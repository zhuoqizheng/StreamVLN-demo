import base64
import importlib.util
import io
import pathlib
import sys

import pytest
from PIL import Image


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
UI_APP_PATH = PROJECT_ROOT / "ui" / "app.py"


def _load_ui_module():
    ui_dir = str(PROJECT_ROOT / "ui")
    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)

    module_name = "ui_app_test_module"
    spec = importlib.util.spec_from_file_location(module_name, UI_APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _jpeg_data_url(width=16, height=12):
    img = Image.new("RGB", (width, height), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@pytest.fixture
def ui_module():
    return _load_ui_module()


def test_data_url_to_jpeg_bytes_valid(ui_module):
    data_url = _jpeg_data_url()
    out = ui_module._data_url_to_jpeg_bytes(data_url, max_size=(8, 8), jpeg_quality=70)
    assert isinstance(out, bytes)
    assert len(out) > 0


def test_data_url_to_jpeg_bytes_invalid(ui_module):
    with pytest.raises(ValueError, match="invalid data url"):
        ui_module._data_url_to_jpeg_bytes("not-a-data-url")


def test_extract_next_waypoint_pixel_multiple_formats(ui_module):
    text_json = """
    ```json
    {"next_waypoint_pixel": {"x": 12, "y": 34}, "reason": "ok"}
    ```
    """
    assert ui_module.extract_next_waypoint_pixel(text_json, 100, 100) == (12, 34)

    text_named = "next_waypoint_pixel: 120, -5"
    assert ui_module.extract_next_waypoint_pixel(text_named, 100, 100) == (99, 0)

    text_tuple = "next point is (7, 8)"
    assert ui_module.extract_next_waypoint_pixel(text_tuple, 100, 100) == (7, 8)


def test_call_vln_eval_fallback_attempts(ui_module, monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, ok, payload=None, status_code=400, text="bad"):
            self.ok = ok
            self._payload = payload or {}
            self.status_code = status_code
            self.text = text

        def json(self):
            return self._payload

    responses = [
        FakeResponse(False, status_code=400, text="format mismatch"),
        FakeResponse(True, payload={"action": [2, 2, 1]}),
    ]

    def fake_post(url, files, data, timeout):
        calls.append({"url": url, "files": files, "data": data, "timeout": timeout})
        return responses[len(calls) - 1]

    monkeypatch.setattr(ui_module.requests, "post", fake_post)

    actions, latency = ui_module._call_vln_eval(
        image_bytes=b"jpeg-bytes",
        reset=True,
        server_url="http://jetson:5801/eval_vln",
        instruction="go",
        timeout_sec=3.0,
    )

    assert actions == [2, 2, 1]
    assert latency >= 0
    assert len(calls) == 2


def test_api_vln_health_success(ui_module, monkeypatch):
    class FakeResponse:
        ok = True

        def json(self):
            return {"step_id": 5, "terminate": False}

    monkeypatch.setattr(ui_module.requests, "get", lambda *_args, **_kwargs: FakeResponse())
    client = ui_module.app.test_client()
    resp = client.get("/api/vln_health?server_url=http://jetson:5801/eval_vln")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["step_id"] == 5
    assert payload["health_url"] == "http://jetson:5801/healthz"


def test_api_vln_health_upstream_error(ui_module, monkeypatch):
    class FakeResponse:
        ok = False
        status_code = 503
        text = "service unavailable"

    monkeypatch.setattr(ui_module.requests, "get", lambda *_args, **_kwargs: FakeResponse())
    client = ui_module.app.test_client()
    resp = client.get("/api/vln_health?server_url=http://jetson:5801/eval_vln")

    assert resp.status_code == 502
    payload = resp.get_json()
    assert payload["ok"] is False
    assert "upstream 503" in payload["error"]
