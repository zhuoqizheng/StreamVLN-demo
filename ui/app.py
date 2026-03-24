from flask import Flask, jsonify, render_template, request
import requests
from PIL import Image, ImageDraw
import io
import base64
import time
import threading
import json
import re
import os

from llm_config import VLM_API_KEY, ENDPOINT, MODEL_NAME, MAX_TOKENS, TEMPERATURE

app = Flask(__name__)

# ================== 配置区 ==================
# Optional legacy source used by index.html single-image VLM page.
SHOT_URL = os.environ.get("UI_SHOT_URL", "http://127.0.0.1:8081/shot.jpg")
VLN_SERVER_URL = os.environ.get("UI_DEFAULT_VLN_SERVER_URL", "http://127.0.0.1:5801/eval_vln")
DEFAULT_VLN_INSTRUCTION = "沿着办公室主要道路行走"

last_prompt = "任务"
# ===========================================

last_request_time = 0
request_lock = threading.Lock()
MIN_INTERVAL_SECONDS = 8


def _data_url_to_jpeg_bytes(data_url, max_size=(960, 540), jpeg_quality=85):
    if not data_url or "," not in data_url:
        raise ValueError("invalid data url")
    encoded = data_url.split(",", 1)[1]
    raw = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail(max_size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


def _call_vln_eval(image_bytes, reset, server_url, instruction="", timeout_sec=30.0):
    request_json = {"reset": bool(reset)}
    if instruction:
        request_json["instruction"] = instruction

    # Different VLN deployments may expect different multipart field names.
    attempts = [
        {
            "files": {"image": ("browser_frame.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            "data": {"json": json.dumps(request_json)},
            "tag": "image+json(reset,instruction)",
        },
        {
            "files": {"file": ("browser_frame.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            "data": {"json": json.dumps(request_json)},
            "tag": "file+json(reset,instruction)",
        },
        {
            "files": {"image": ("browser_frame.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            "data": {
                "reset": str(bool(reset)).lower(),
                "instruction": instruction,
            },
            "tag": "image+form(reset,instruction)",
        },
        {
            "files": {"image": ("browser_frame.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            "data": {
                "reset": str(bool(reset)).lower(),
                "instruction_text": instruction,
            },
            "tag": "image+form(reset,instruction_text)",
        },
    ]

    last_error = None
    for attempt in attempts:
        t0 = time.time()
        resp = requests.post(
            server_url,
            files=attempt["files"],
            data=attempt["data"],
            timeout=timeout_sec,
        )
        latency = time.time() - t0

        if resp.ok:
            payload = resp.json()
            actions = payload.get("action", [])
            if not isinstance(actions, list):
                actions = list(actions)
            return actions, latency

        body_preview = (resp.text or "").strip().replace("\n", " ")[:300]
        last_error = (
            f"upstream {resp.status_code} via {attempt['tag']}: "
            f"{body_preview or 'no response body'}"
        )

    raise RuntimeError(last_error or "all upstream formats failed")


def extract_next_waypoint_pixel(model_text, width, height):
    if not model_text:
        return None

    text = model_text.strip()
    candidates = [text]
    json_fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if json_fenced:
        candidates.insert(0, json_fenced.group(1).strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            point = data.get("next_waypoint_pixel", {}) if isinstance(data, dict) else {}
            x = int(point.get("x"))
            y = int(point.get("y"))
            return max(0, min(x, width - 1)), max(0, min(y, height - 1))
        except Exception:
            pass

    named_match = re.search(
        r"next_waypoint_pixel[^\d-]*(-?\d+)\s*[,，]\s*(-?\d+)",
        text,
        re.IGNORECASE,
    )
    if named_match:
        x = int(named_match.group(1))
        y = int(named_match.group(2))
        return max(0, min(x, width - 1)), max(0, min(y, height - 1))

    tuple_match = re.search(r"[\(\[]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[\)\]]", text)
    if tuple_match:
        x = int(tuple_match.group(1))
        y = int(tuple_match.group(2))
        return max(0, min(x, width - 1)), max(0, min(y, height - 1))

    return None


def draw_waypoint_dot(image, waypoint):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    x, y = waypoint
    radius = 8

    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red", outline="white", width=2)
    draw.line((x - 14, y, x + 14, y), fill="red", width=2)
    draw.line((x, y - 14, x, y + 14), fill="red", width=2)
    return marked


@app.route('/', methods=['GET', 'POST'])
def index():
    global last_prompt, last_request_time

    result = None
    image_b64 = None
    waypoint_image_b64 = None
    error_msg = None

    if request.method == 'POST':
        user_prompt = request.form.get('prompt', '').strip()
        use_last = request.form.get('use_last') == 'true'

        if use_last or not user_prompt:
            user_prompt = last_prompt
        else:
            last_prompt = user_prompt

        current_time = time.time()
        with request_lock:
            if current_time - last_request_time < MIN_INTERVAL_SECONDS:
                error_msg = f"请求过于频繁，请等待 {int(MIN_INTERVAL_SECONDS - (current_time - last_request_time))} 秒后再试（保护额度机制）"
            else:
                last_request_time = current_time

        if error_msg:
            return render_template('index.html', result=result, image_b64=image_b64,
                                   waypoint_image_b64=waypoint_image_b64,
                                   last_prompt=last_prompt, error=error_msg)

        # 1. 采集图像
        try:
            resp = requests.get(SHOT_URL, timeout=8)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            img = img.resize((512, 384))
        except Exception as e:
            error_msg = (
                f"图像源连接失败：{str(e)}<br>"
                f"请确认 UI_SHOT_URL 指向可访问的单帧 JPEG 接口（当前: {SHOT_URL}）"
            )
            return render_template('index.html', result=result, image_b64=image_b64,
                                   waypoint_image_b64=waypoint_image_b64,
                                   last_prompt=last_prompt, error=error_msg)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=75)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_b64 = img_base64

        # 2. 准备 prompt
        full_prompt = (
            "你是一个机器人导航专家。分析下面这张摄像头图像。\n"
            f"用户指令：{user_prompt}\n\n"
            "请返回机器人下一步需要到达的图像像素坐标。\n"
            "如果不需要移动请输出需要转向的方向和角度。\n"
            "你必须包含简单的场景分析和输出 JSON，格式严格如下：\n"
            "{\"next_waypoint_pixel\": {\"x\": 整数, \"y\": 整数}, \"是否转向\": \"转向角度\", \"reason\": \"一句话中文理由\"}\n"
            "要求：x 范围 [0, 511]，y 范围 [0, 383]"
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }

        if not VLM_API_KEY:
            error_msg = "未配置 VLM_API_KEY。请先在 llm_config.py 中设置后再运行。"
            return render_template('index.html', result=result, image_b64=image_b64,
                                   waypoint_image_b64=waypoint_image_b64,
                                   last_prompt=last_prompt, error=error_msg)

        headers = {
            "Authorization": f"Bearer {VLM_API_KEY}",
            "Content-Type": "application/json"
        }

        # 3. 调用大模型，带重试
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(ENDPOINT, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                result = response.json()["choices"][0]["message"]["content"].strip()

                waypoint = extract_next_waypoint_pixel(result, width=img.width, height=img.height)
                if waypoint:
                    x, y = waypoint
                    marked_img = draw_waypoint_dot(img, waypoint)
                    marked_buffer = io.BytesIO()
                    marked_img.save(marked_buffer, format="JPEG", quality=85)
                    waypoint_image_b64 = base64.b64encode(marked_buffer.getvalue()).decode('utf-8')
                    result = f"{result}\n\n解析到下一个路径点像素坐标：({x}, {y})"
                else:
                    result = f"{result}\n\n未能从模型输出中解析到像素坐标，请检查返回格式。"
                break
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else 0
                if status == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                elif status in (401, 403):
                    result = f"认证/额度错误（{status}）：请检查 API Key 或控制台免费额度是否用尽"
                    break
                else:
                    result = f"API 调用失败（{status}）：{str(e)}"
                    break
            except Exception as e:
                result = f"请求异常：{str(e)}"
                break
        else:
            result = "达到最大重试次数，建议等待几分钟或检查控制台。"

    return render_template('index.html', result=result, image_b64=image_b64,
                           waypoint_image_b64=waypoint_image_b64,
                           last_prompt=last_prompt, error=error_msg)


@app.route('/vln_web')
def vln_web():
    return render_template(
        'vln_web.html',
        default_server_url=VLN_SERVER_URL,
        default_instruction=DEFAULT_VLN_INSTRUCTION,
    )


@app.route('/api/eval_vln', methods=['POST'])
def api_eval_vln():
    body = request.get_json(silent=True) or {}
    image_data_url = body.get('image_data_url', '')
    reset = bool(body.get('reset', False))
    instruction = (body.get('instruction') or '').strip()
    timeout = float(body.get('timeout', 30.0))
    server_url = (body.get('server_url') or VLN_SERVER_URL).strip()

    if not image_data_url:
        return jsonify({"ok": False, "error": "missing image_data_url"}), 400

    try:
        image_bytes = _data_url_to_jpeg_bytes(image_data_url)
    except Exception as e:
        return jsonify({"ok": False, "error": f"invalid image: {e}"}), 400

    try:
        actions, latency = _call_vln_eval(
            image_bytes=image_bytes,
            reset=reset,
            server_url=server_url,
            instruction=instruction,
            timeout_sec=timeout,
        )
        return jsonify({
            "ok": True,
            "actions": actions,
            "latency": latency,
            "server_url": server_url,
            "instruction": instruction,
        })
    except requests.exceptions.RequestException as e:
        body_preview = ""
        if hasattr(e, "response") and e.response is not None:
            body_preview = (e.response.text or "").strip().replace("\n", " ")[:300]
        msg = f"request failed: {e}"
        if body_preview:
            msg = f"{msg} | upstream body: {body_preview}"
        return jsonify({"ok": False, "error": msg}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/vln_health', methods=['GET'])
def api_vln_health():
    server_url = (request.args.get('server_url') or VLN_SERVER_URL).strip()
    base = server_url[:-len('/eval_vln')] if server_url.endswith('/eval_vln') else server_url
    health_url = base.rstrip('/') + '/healthz'

    try:
        t0 = time.time()
        resp = requests.get(health_url, timeout=5)
        latency = time.time() - t0
        if not resp.ok:
            body_preview = (resp.text or '').strip().replace('\n', ' ')[:300]
            return jsonify({
                'ok': False,
                'health_url': health_url,
                'error': f'upstream {resp.status_code}: {body_preview or "no body"}',
            }), 502

        payload = resp.json()
        payload['ok'] = True
        payload['health_url'] = health_url
        payload['latency'] = latency
        return jsonify(payload)
    except requests.exceptions.RequestException as e:
        return jsonify({
            'ok': False,
            'health_url': health_url,
            'error': str(e),
        }), 502


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
