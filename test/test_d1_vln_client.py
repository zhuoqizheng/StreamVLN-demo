import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
D1_CLIENT_PATH = PROJECT_ROOT / "realworld" / "d1_vln_client.py"


def _load_d1_client_module():
    # Stub ROS-related modules so this test can run without robot runtime dependencies.
    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.init = lambda: None
    rclpy_mod.shutdown = lambda: None
    rclpy_mod.spin = lambda *_args, **_kwargs: None

    rclpy_node_mod = types.ModuleType("rclpy.node")

    class DummyNode:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_subscription(self, *_args, **_kwargs):
            return None

        def get_logger(self):
            class _Logger:
                def info(self, *_args, **_kwargs):
                    return None

                def warning(self, *_args, **_kwargs):
                    return None

            return _Logger()

    rclpy_node_mod.Node = DummyNode

    cv_bridge_mod = types.ModuleType("cv_bridge")

    class DummyCvBridge:
        def imgmsg_to_cv2(self, *_args, **_kwargs):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    cv_bridge_mod.CvBridge = DummyCvBridge

    sensor_msgs_mod = types.ModuleType("sensor_msgs")
    sensor_msgs_msg_mod = types.ModuleType("sensor_msgs.msg")

    class DummyImage:
        pass

    sensor_msgs_msg_mod.Image = DummyImage

    sys.modules["rclpy"] = rclpy_mod
    sys.modules["rclpy.node"] = rclpy_node_mod
    sys.modules["cv_bridge"] = cv_bridge_mod
    sys.modules["sensor_msgs"] = sensor_msgs_mod
    sys.modules["sensor_msgs.msg"] = sensor_msgs_msg_mod

    realworld_dir = str(PROJECT_ROOT / "realworld")
    if realworld_dir not in sys.path:
        sys.path.insert(0, realworld_dir)

    module_name = "d1_vln_client_test_module"
    spec = importlib.util.spec_from_file_location(module_name, D1_CLIENT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def d1_module(monkeypatch):
    module = _load_d1_client_module()
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)
    module.policy_init = True
    return module


def test_eval_vln_success_and_reset_toggle(d1_module, monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, files, data, timeout):
        calls.append({"url": url, "data": data, "timeout": timeout, "files": files})
        return FakeResponse({"action": [1, 2, 3]})

    monkeypatch.setattr(d1_module.requests, "post", fake_post)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    first_actions = d1_module.eval_vln(image, instruction="go forward", url="http://server/eval_vln")
    second_actions = d1_module.eval_vln(image, instruction="go forward", url="http://server/eval_vln")

    assert first_actions == [1, 2, 3]
    assert second_actions == [1, 2, 3]
    assert len(calls) == 2

    first_payload = json.loads(calls[0]["data"]["json"])
    second_payload = json.loads(calls[1]["data"]["json"])
    assert first_payload["reset"] is True
    assert second_payload["reset"] is False
    assert first_payload["instruction"] == "go forward"


def test_eval_vln_retries_then_fails(d1_module, monkeypatch):
    call_count = {"n": 0}

    def always_fail(*_args, **_kwargs):
        call_count["n"] += 1
        raise RuntimeError("network down")

    monkeypatch.setenv("STREAMVLN_RETRY", "1")
    monkeypatch.setattr(d1_module.requests, "post", always_fail)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="eval_vln request failed"):
        d1_module.eval_vln(image, instruction="x", url="http://server/eval_vln")

    assert call_count["n"] == 2


def test_normalize_arch_maps_common_values(d1_module, monkeypatch):
    monkeypatch.setattr(d1_module.platform, "machine", lambda: "amd64")
    assert d1_module._normalize_arch() == "x86_64"

    monkeypatch.setattr(d1_module.platform, "machine", lambda: "arm64")
    assert d1_module._normalize_arch() == "aarch64"


def test_load_sdk_module_rejects_invalid_model(d1_module, monkeypatch):
    monkeypatch.setenv("D1_MODEL", "invalid-model")
    with pytest.raises(ValueError, match="D1_MODEL must be one of"):
        d1_module._load_d1_sdk_module()


def test_incremental_change_goal_updates_pose(d1_module):
    class DummyManager:
        def __init__(self):
            self.homo_goal = np.eye(4)

    mgr = DummyManager()

    d1_module.D1VlnManager.incremental_change_goal(mgr, [1])
    assert np.isclose(mgr.homo_goal[0, 3], 0.25)
    assert np.isclose(mgr.homo_goal[1, 3], 0.0)

    mgr.homo_goal = np.eye(4)
    d1_module.D1VlnManager.incremental_change_goal(mgr, [2, 3])
    assert np.allclose(mgr.homo_goal[:3, :3], np.eye(3), atol=1e-6)


def test_eval_vln_with_sample_image_data(d1_module, monkeypatch, tmp_path):
    sample_path = tmp_path / "sample_rgb.jpg"
    sample_rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    sample_rgb[..., 0] = 220
    sample_rgb[..., 1] = np.tile(np.arange(48, dtype=np.uint8), (32, 1))
    sample_rgb[..., 2] = np.tile(np.arange(32, dtype=np.uint8).reshape(-1, 1), (1, 48))
    Image.fromarray(sample_rgb).save(sample_path)

    image = np.array(Image.open(sample_path).convert("RGB"))
    captured = {"json": None, "jpeg_header": None}

    class FakeResponse:
        text = '{"action": [1, 1, 3, 0]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"action": [1, 1, 3, 0]}

    def fake_post(_url, files, data, timeout):
        _name, file_obj, _mime = files["image"]
        file_obj.seek(0)
        payload = file_obj.read()
        captured["jpeg_header"] = payload[:2]
        captured["json"] = json.loads(data["json"])
        assert timeout > 0
        return FakeResponse()

    monkeypatch.setenv("STREAMVLN_INSTRUCTION", "沿主走廊前进")
    monkeypatch.setattr(d1_module.requests, "post", fake_post)

    actions = d1_module.eval_vln(image, instruction=None, url="http://server/eval_vln")
    assert actions == [1, 1, 3, 0]
    assert captured["json"]["instruction"] == "沿主走廊前进"
    assert captured["jpeg_header"] == b"\xff\xd8"
