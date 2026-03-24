import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from io import BytesIO
from urllib import error, request as urllib_request

import numpy as np
import torch
import transformers
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamvln.streamvln_agent import VLNEvaluator
from model.stream_video_vln import StreamVLNForCausalLM


app = Flask(__name__)
action_seq = np.zeros(4)
idx = 0
terminate = False
total_generate_time = 0.0
start_time = time.time()
output_dir = ""
logger = logging.getLogger("http_realworld_server_2")
last_task_result = None


def extract_task_from_instruction(instruction: str):
    if not instruction:
        return "", None

    match = re.search(r"任务\s*[：:]\s*(.+)$", instruction, flags=re.DOTALL)
    if not match:
        return instruction.strip(), None

    task_text = match.group(1).strip()
    navigation_text = instruction[:match.start()].strip()
    if not navigation_text:
        navigation_text = instruction.strip()
    return navigation_text, task_text or None


def pil_image_to_data_url(image: Image.Image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def parse_qwen_message_content(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("output_text")
                if text:
                    parts.append(str(text).strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return "\n".join(part for part in parts if part)
    return str(content).strip()


def analyze_task_with_qwen_vl(task_text: str, image_rgb: Image.Image):
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY or QWEN_API_KEY for online Qwen-VL analysis.")

    api_url = app.config.get("qwen_vl_api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    model_name = app.config.get("qwen_vl_model", "qwen2.5-vl-72b-instruct")
    prompt = (
        "你是一个视觉检查助手。请基于当前图片完成任务判断，并给出简洁结论。"
        f"\n任务：{task_text}"
        "\n请输出 JSON 字符串，包含字段："
        "result(通过/不通过/不确定)、summary(一句话结论)、evidence(观察依据)。"
    )
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": pil_image_to_data_url(image_rgb)}},
                ],
            }
        ],
        "temperature": 0.1,
    }

    req = urllib_request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=60) as response:
            resp_data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Qwen-VL API HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Qwen-VL API connection failed: {exc}") from exc

    choices = resp_data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Qwen-VL API returned no choices: {resp_data}")

    message = choices[0].get("message", {})
    return parse_qwen_message_content(message.get("content", ""))


def setup_logger(log_path: str):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def append_jsonl(log_jsonl_path: str, data: dict):
    with open(log_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def annotate_image(frame_idx, image_bgr, run_start_time, gen_time, action_text, save_dir):
    image = Image.fromarray(image_bgr[..., ::-1])
    draw = ImageDraw.Draw(image)
    font_size = 20
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    text_content = [
        f"Frame    Id  : {frame_idx}",
        f"Running  time: {time.time() - run_start_time:.2f} s",
        f"Generate time: {gen_time:.2f} s",
        f"Actions      : {action_text}",
    ]

    max_width = 0
    total_height = 0
    for line in text_content:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = 26
        max_width = max(max_width, text_width)
        total_height += text_height

    padding = 10
    box_x, box_y = 10, 10
    box_width = max_width + 2 * padding
    box_height = total_height + 2 * padding
    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill="black")

    y_position = box_y + padding
    for line in text_content:
        draw.text((box_x + padding, y_position), line, fill="white", font=font)
        y_position += 26

    raw_path = os.path.join(save_dir, f"rgb_{frame_idx:06d}.png")
    ann_path = os.path.join(save_dir, f"rgb_{frame_idx:06d}_annotated.png")

    Image.fromarray(image_bgr[..., ::-1]).save(raw_path)
    image.save(ann_path)
    return raw_path, ann_path


@app.route("/eval_vln", methods=["POST"])
def eval_vln():
    global action_seq, idx, terminate, total_generate_time, output_dir, start_time, last_task_result

    req_start = time.time()
    image_file = request.files["image"]
    json_data = request.form["json"]
    data = json.loads(json_data)

    image_rgb = Image.open(image_file.stream).convert("RGB")
    image = np.asarray(image_rgb)[..., ::-1]  # BGR

    instruction = data.get("instruction", "Walk forward and immediately stop when you exit the room.")
    navigation_instruction, task_text = extract_task_from_instruction(instruction)
    policy_init = bool(data.get("reset", False))

    if policy_init:
        start_time = time.time()
        total_generate_time = 0.0
        terminate = False
        idx = 0
        last_task_result = None

        session_name = "runs_" + datetime.now().strftime("%m-%d-%H%M%S")
        output_dir = os.path.join(app.config["save_root"], session_name)
        os.makedirs(output_dir, exist_ok=True)

        setup_logger(os.path.join(output_dir, "server.log"))
        logger.info("reset=true, new session: %s", output_dir)
        evaluator.reset_memory()

    if not output_dir:
        output_dir = os.path.join(app.config["save_root"], "runs_default")
        os.makedirs(output_dir, exist_ok=True)
        setup_logger(os.path.join(output_dir, "server.log"))

    idx += 1

    if terminate:
        logger.info("task already terminated, return STOP")
        return jsonify({"action": [0], "task_result": last_task_result})

    llm_output = ""
    for _ in range(4):
        t1 = time.time()
        return_action, generate_time, return_llm_output = evaluator.step(
            0,
            image,
            navigation_instruction,
            run_model=(evaluator.step_id % 4 == 0),
        )
        logger.info("one evaluate cost %.3f s", time.time() - t1)

        if return_llm_output is not None:
            llm_output = return_llm_output
        if generate_time > 0:
            total_generate_time = generate_time
        action_seq = action_seq if return_action is None else return_action

        if 0 in action_seq:
            terminate = True
        evaluator.step_id += 1

    str_action = [str(i) for i in action_seq]
    str_action = "".join(str_action)
    str_action = str_action.replace("1", "↑")
    str_action = str_action.replace("2", "←")
    str_action = str_action.replace("3", "→")
    str_action = str_action.replace("0", "STOP")

    if idx > 1 and total_generate_time > 0.5:
        total_generate_time -= 0.3

    raw_path, ann_path = annotate_image(idx, image, start_time, total_generate_time, str_action, output_dir)

    task_result = None
    action_values = action_seq.tolist() if hasattr(action_seq, "tolist") else list(action_seq)
    task_finished = bool(action_values) and action_values[0] == 0
    if task_text and task_finished:
        try:
            task_result = analyze_task_with_qwen_vl(task_text, image_rgb)
            last_task_result = {
                "task": task_text,
                "result": task_result,
            }
            logger.info("task analysis success: %s", task_result)
        except Exception as exc:
            task_result = f"Qwen-VL analysis failed: {exc}"
            last_task_result = {
                "task": task_text,
                "result": task_result,
            }
            logger.exception("task analysis failed")

    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "idx": idx,
        "reset": policy_init,
        "instruction": instruction,
        "navigation_instruction": navigation_instruction,
        "task": task_text,
        "action": action_values,
        "action_text": str_action,
        "llm_output": llm_output,
        "generate_time": float(total_generate_time),
        "request_time": float(time.time() - req_start),
        "terminate": bool(terminate),
        "raw_image": raw_path,
        "annotated_image": ann_path,
        "task_result": task_result,
    }
    append_jsonl(os.path.join(output_dir, "events.jsonl"), event)
    logger.info("idx=%s action=%s terminate=%s", idx, event["action"], terminate)

    if len(action_seq) == 0:
        logger.info("empty action, return STOP")
        return jsonify({"action": [0], "task_result": last_task_result})

    return jsonify({"action": event["action"], "task_result": last_task_result if task_finished else None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/tf/StreamVLN/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=4096,
        help="Maximum sequence length. Sequences will be right padded (and possibly truncated).",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="device to use for testing")
    parser.add_argument("--save_root", type=str, default="/tf/StreamVLN/data_image")
    parser.add_argument("--port", type=int, default=5802)
    parser.add_argument("--qwen_vl_model", type=str, default="qwen2.5-vl-72b-instruct")
    parser.add_argument("--qwen_vl_api_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

    args = parser.parse_args()

    if isinstance(args.device, str) and args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] Requested device '{args.device}' but CUDA is unavailable. Falling back to CPU.")
        args.device = "cpu"

    model_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    os.makedirs(args.save_root, exist_ok=True)
    app.config["save_root"] = args.save_root
    app.config["qwen_vl_model"] = args.qwen_vl_model
    app.config["qwen_vl_api_url"] = args.qwen_vl_api_url

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )

    config = transformers.AutoConfig.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="sdpa",
        torch_dtype=model_dtype,
        config=config,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.model.num_history = args.num_history
    model.reset(1)
    model.requires_grad_(False)
    model.to(args.device)
    model.eval()

    vln_sensor_config = {
        "rgb_height": 1.25,
        "camera_intrinsic": np.array(
            [
                [192.0, 0.0, 191.42857143, 0.0],
                [0.0, 192.0, 191.42857143, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }

    evaluator = VLNEvaluator(
        vln_sensor_config,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )

    # warmup
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=True)
    evaluator.reset_memory()

    app.run(host="0.0.0.0", port=args.port)
