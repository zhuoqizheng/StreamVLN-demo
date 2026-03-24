import importlib.util
import io
import json
import pathlib
import sys
import types

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER_PATH = PROJECT_ROOT / "streamvln" / "http_realworld_server.py"


def _load_server_module():
    # Stub heavy model imports so route/unit tests can run in isolation.
    streamvln_agent_mod = types.ModuleType("streamvln.streamvln_agent")

    class DummyEvaluator:
        def __init__(self, *_args, **_kwargs):
            self.step_id = 0

    streamvln_agent_mod.VLNEvaluator = DummyEvaluator

    model_pkg = types.ModuleType("model")
    model_stream_mod = types.ModuleType("model.stream_video_vln")

    class DummyModel:
        pass

    model_stream_mod.StreamVLNForCausalLM = DummyModel

    sys.modules["streamvln.streamvln_agent"] = streamvln_agent_mod
    sys.modules["model"] = model_pkg
    sys.modules["model.stream_video_vln"] = model_stream_mod

    module_name = "http_realworld_server_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_jpeg_bytes():
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def server_module(monkeypatch):
    module = _load_server_module()

    class FakeEvaluator:
        def __init__(self):
            self.step_id = 0
            self.reset_called = False

        def reset_memory(self):
            self.reset_called = True

        def step(self, *_args, **_kwargs):
            return np.array([1, 2, 3, 0]), 0.8, "actions"

    module.evaluator = FakeEvaluator()
    monkeypatch.setattr(module, "annotate_image", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.os, "makedirs", lambda *_args, **_kwargs: None)

    module.action_seq = np.zeros(4)
    module.idx = 0
    module.terminate = False
    module.total_generate_time = 0.0
    return module


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("yes", True),
        ("0", False),
        (1, True),
        (0, False),
        (None, False),
    ],
)
def test_parse_bool(server_module, raw, expected):
    assert server_module._parse_bool(raw) is expected


def test_to_action_list(server_module):
    assert server_module._to_action_list(None) == [0]
    assert server_module._to_action_list(np.array([1, 2])) == [1, 2]
    assert server_module._to_action_list([3, 4]) == [3, 4]


def test_eval_vln_rejects_missing_image(server_module):
    client = server_module.app.test_client()
    resp = client.post("/eval_vln", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "missing multipart image field" in resp.get_json()["error"]


def test_eval_vln_rejects_invalid_json_field(server_module):
    client = server_module.app.test_client()
    data = {
        "image": (io.BytesIO(_make_jpeg_bytes()), "frame.jpg"),
        "json": "{bad-json",
    }
    resp = client.post("/eval_vln", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "invalid json field" in resp.get_json()["error"]


def test_eval_vln_success(server_module):
    client = server_module.app.test_client()
    data = {
        "image": (io.BytesIO(_make_jpeg_bytes()), "frame.jpg"),
        "json": json.dumps({"reset": True, "instruction": "沿走廊前进"}),
    }

    resp = client.post("/eval_vln", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    payload = resp.get_json()

    assert payload["action"] == [1, 2, 3, 0]
    assert payload["instruction"] == "沿走廊前进"
    assert payload["frame_id"] == 1
    assert payload["terminate"] is True
    assert server_module.evaluator.reset_called is True


def test_healthz(server_module):
    client = server_module.app.test_client()
    server_module.evaluator.step_id = 12
    server_module.terminate = True

    resp = client.get("/healthz")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["step_id"] == 12
    assert payload["terminate"] is True
