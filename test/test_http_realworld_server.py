import importlib.util
import io
import json
import logging
import pathlib
import sys
import types

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER_PATH = PROJECT_ROOT / "streamvln" / "http_realworld_server.py"
LOGGER = logging.getLogger(__name__)


def _load_server_module():
    LOGGER.info("Loading module from %s", SERVER_PATH)
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
    LOGGER.debug("Creating synthetic JPEG bytes for upload")
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def server_module(monkeypatch):
    LOGGER.info("Preparing server fixture with mocked evaluator")
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
    LOGGER.info("Server fixture ready")
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
    LOGGER.debug("Asserting _parse_bool(%r) == %r", raw, expected)
    assert server_module._parse_bool(raw) is expected


def test_to_action_list(server_module):
    LOGGER.info("Testing _to_action_list conversions")
    assert server_module._to_action_list(None) == [0]
    assert server_module._to_action_list(np.array([1, 2])) == [1, 2]
    assert server_module._to_action_list([3, 4]) == [3, 4]


def test_eval_vln_rejects_missing_image(server_module):
    client = server_module.app.test_client()
    resp = client.post("/eval_vln", data={}, content_type="multipart/form-data")
    LOGGER.info("Missing image response status=%s body=%s", resp.status_code, resp.get_json())
    assert resp.status_code == 400
    assert "missing multipart image field" in resp.get_json()["error"]


def test_eval_vln_rejects_invalid_json_field(server_module):
    client = server_module.app.test_client()
    data = {
        "image": (io.BytesIO(_make_jpeg_bytes()), "frame.jpg"),
        "json": "{bad-json",
    }
    resp = client.post("/eval_vln", data=data, content_type="multipart/form-data")
    LOGGER.info("Invalid json response status=%s body=%s", resp.status_code, resp.get_json())
    assert resp.status_code == 400
    assert "invalid json field" in resp.get_json()["error"]


def test_eval_vln_success(server_module):
    client = server_module.app.test_client()
    data = {
        "image": (io.BytesIO(_make_jpeg_bytes()), "frame.jpg"),
        "json": json.dumps({"reset": True, "instruction": "沿走廊前进"}),
    }

    resp = client.post("/eval_vln", data=data, content_type="multipart/form-data")
    LOGGER.info("Success response status=%s", resp.status_code)
    assert resp.status_code == 200
    payload = resp.get_json()
    LOGGER.info("Success response payload=%s", payload)

    assert payload["action"] == [1, 2, 3, 0]
    assert payload["instruction"] == "沿走廊前进"
    assert payload["frame_id"] == 1
    assert payload["terminate"] is True
    assert server_module.evaluator.reset_called is True


def test_eval_vln_with_sample_image_file_field_and_plain_form(server_module, tmp_path):
    # Build a richer sample image to emulate a camera frame.
    sample_path = tmp_path / "sample_frame.jpg"
    sample = np.zeros((36, 64, 3), dtype=np.uint8)
    sample[..., 0] = 120
    sample[..., 1] = np.tile(np.arange(64, dtype=np.uint8), (36, 1))
    sample[..., 2] = np.tile(np.arange(36, dtype=np.uint8).reshape(-1, 1), (1, 64))
    Image.fromarray(sample).save(sample_path)

    client = server_module.app.test_client()
    with sample_path.open("rb") as f:
        data = {
            "file": (io.BytesIO(f.read()), "sample_frame.jpg"),
            "reset": "true",
            "instruction_text": "沿着走廊前进后左转",
        }
        resp = client.post("/eval_vln", data=data, content_type="multipart/form-data")

    LOGGER.info("Sample-image response status=%s", resp.status_code)
    assert resp.status_code == 200
    payload = resp.get_json()
    LOGGER.info("Sample-image response payload=%s", payload)
    assert payload["action"] == [1, 2, 3, 0]
    assert payload["instruction"] == "沿着走廊前进后左转"
    assert payload["frame_id"] == 1
    assert payload["terminate"] is True
    assert server_module.evaluator.reset_called is True


def test_healthz(server_module):
    client = server_module.app.test_client()
    server_module.evaluator.step_id = 12
    server_module.terminate = True

    resp = client.get("/healthz")
    LOGGER.info("Healthz response status=%s", resp.status_code)
    assert resp.status_code == 200
    payload = resp.get_json()
    LOGGER.info("Healthz response payload=%s", payload)
    assert payload["ok"] is True
    assert payload["step_id"] == 12
    assert payload["terminate"] is True


if __name__ == "__main__":
    import pytest

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    raise SystemExit(pytest.main(["-vv", "-s", "--log-cli-level=INFO", __file__]))
