#!/usr/bin/env python3
import argparse
import io
import json
import time
from typing import List, Tuple

import cv2
import numpy as np
import requests
from PIL import Image as PIL_Image


ACTION_NAME = {
    0: "STOP",
    1: "FORWARD",
    2: "LEFT",
    3: "RIGHT",
}


def eval_vln(
    image_bgr: np.ndarray,
    server_url: str,
    policy_init: bool,
    instruction: str = "Walk forward and immediately stop when you exit the room.",
    timeout_sec: float = 30.0,
) -> Tuple[List[int], str, str, float]:
    """
    Compatible with server endpoint /eval_vln:
    - multipart file field: image
    - form field: json, with {"reset": bool, "instruction": str}
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PIL_Image.fromarray(image_rgb)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    data = {"reset": policy_init, "instruction": instruction}
    files = {"image": ("iphone_frame.jpg", buf, "image/jpeg")}

    t0 = time.time()
    resp = requests.post(
        server_url,
        files=files,
        data={"json": json.dumps(data)},
        timeout=timeout_sec,
    )
    latency = time.time() - t0
    resp.raise_for_status()

    payload = resp.json()
    actions = payload.get("action", [])
    llm_output = payload.get("llm_output", "")
    status = payload.get("status", "not_provided")
    if not isinstance(actions, list):
        actions = list(actions)
    return actions, llm_output, status, latency


def actions_to_text(actions: List[int]) -> str:
    if len(actions) == 0:
        return "[]"
    names = [ACTION_NAME.get(int(a), f"UNK({a})") for a in actions]
    return " -> ".join(names)


def draw_overlay(frame: np.ndarray, text_lines: List[str]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    line_h = 28
    box_h = line_h * len(text_lines) + 20
    box_w = min(w - 20, 900)

    cv2.rectangle(out, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)

    y = 35
    for line in text_lines:
        cv2.putText(
            out,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += line_h
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iphone_stream",
        type=str,
        required=True,
        help="iPhone stream URL, e.g. http://192.168.1.20:8080/video",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://127.0.0.1:5801/eval_vln",
        help="StreamVLN server endpoint",
    )
    parser.add_argument(
        "--call_interval",
        type=float,
        default=0.2,
        help="Seconds between eval_vln calls",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout seconds for eval_vln",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Walk forward and immediately stop when you exit the room.",
        help="Navigation instruction sent to the server",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.iphone_stream)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open iPhone stream: {args.iphone_stream}")

    policy_init = True
    last_call_ts = 0.0
    last_actions: List[int] = []
    last_llm_output = ""
    last_status = "not_provided"
    last_latency = 0.0
    frame_id = 0

    print("Press 'r' to reset policy_init=True")
    print("Press 'q' to quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Failed to read frame from iPhone stream")
            time.sleep(0.05)
            continue

        frame_id += 1
        now = time.time()

        if now - last_call_ts >= args.call_interval:
            try:
                actions, llm_output, status, latency = eval_vln(
                    frame,
                    server_url=args.server_url,
                    policy_init=policy_init,
                    instruction=args.instruction,
                    timeout_sec=args.timeout,
                )
                policy_init = False
                last_actions = actions
                last_llm_output = llm_output
                last_status = status
                last_latency = latency
                last_call_ts = now

                print(
                    f"[frame {frame_id}] actions={actions} ({actions_to_text(actions)}), "
                    f"status={status}, latency={latency:.3f}s"
                )
                if llm_output:
                    print(f"[frame {frame_id}] llm_output={llm_output}")
            except Exception as e:
                print(f"[frame {frame_id}] eval_vln failed: {e}")

        llm_short = last_llm_output if len(last_llm_output) <= 80 else last_llm_output[:77] + "..."

        vis = draw_overlay(
            frame,
            [
                f"Instruction: {args.instruction}",
                f"Actions: {actions_to_text(last_actions)}",
                f"Status: {last_status}",
                f"LLM: {llm_short}",
                f"Latency: {last_latency:.3f}s",
                f"Frame: {frame_id}",
                "Hotkeys: r=reset, q=quit",
            ],
        )

        cv2.imshow("StreamVLN iPhone Client", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            policy_init = True
            print("policy_init reset to True")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()