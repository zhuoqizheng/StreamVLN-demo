#!/usr/bin/env python3
import argparse
import io
import json
import time
from typing import Any, Dict, List, Tuple

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


def truncate_text(text: str, max_len: int = 120) -> str:
    if text is None:
        return ""
    normalized = str(text).replace("\n", " ").strip()
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


def _parse_actions(payload: Dict[str, Any]) -> List[int]:
    raw = payload.get("action")
    if raw is None:
        raw = payload.get("actions", [])

    if isinstance(raw, (int, float, str)):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("response format invalid: action/actions is not a list")

    out: List[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except Exception as exc:
            raise ValueError(f"response format invalid: non-integer action {item}") from exc
    return out


def _parse_response(payload: Dict[str, Any]) -> Tuple[List[int], str, str, str]:
    if not isinstance(payload, dict):
        raise ValueError("response format invalid: top-level JSON must be an object")

    actions = _parse_actions(payload)
    status = str(payload.get("status", ""))
    analysis = str(payload.get("analysis", ""))
    llm_output = str(payload.get("llm_output", payload.get("output", "")))
    return actions, status, analysis, llm_output


def eval_vln(
    image_bgr: np.ndarray,
    server_url: str,
    instruction: str,
    policy_init: bool,
    timeout_sec: float = 30.0,
) -> Tuple[List[int], str, str, str, float]:
    """Call /eval_vln with image + json(reset,instruction) and parse response fields."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PIL_Image.fromarray(image_rgb)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    req_data = {"reset": bool(policy_init), "instruction": instruction}
    files = {"image": ("iphone_frame.jpg", buf, "image/jpeg")}

    t0 = time.time()
    resp = requests.post(
        server_url,
        files=files,
        data={"json": json.dumps(req_data)},
        timeout=timeout_sec,
    )
    latency = time.time() - t0
    resp.raise_for_status()

    try:
        payload = resp.json()
    except Exception as exc:
        raise ValueError("response format invalid: body is not valid JSON") from exc

    actions, status, analysis, llm_output = _parse_response(payload)
    return actions, status, analysis, llm_output, latency


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
    box_w = min(w - 20, 1180)

    cv2.rectangle(out, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)

    y = 35
    for line in text_lines:
        cv2.putText(
            out,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += line_h
    return out


def main() -> None:
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
        default="http://192.168.51.172:5801/eval_vln",
        help="StreamVLN server endpoint",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Go forward a short distance and avoid obstacles.",
        help="Navigation instruction sent to server",
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
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.iphone_stream)
    camera_desc = f"iPhone stream: {args.iphone_stream}"
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source ({camera_desc})")

    policy_init = True
    last_call_ts = 0.0
    last_actions: List[int] = []
    last_status = ""
    last_analysis = ""
    last_llm_output = ""
    last_latency = 0.0
    frame_id = 0
    read_fail_count = 0

    print(f"Using camera source: {camera_desc}")
    print(f"Server: {args.server_url}")
    print(f"Instruction: {args.instruction}")
    print("Press 'r' to reset policy_init=True")
    print("Press 'q' to quit")

    while True:
        ok, frame = cap.read()
        frame_id += 1
        if not ok or frame is None:
            read_fail_count += 1
            print(f"[frame {frame_id}] image read failed (count={read_fail_count})")
            time.sleep(0.05)
            continue

        read_fail_count = 0
        now = time.time()

        if now - last_call_ts >= args.call_interval:
            try:
                actions, status, analysis, llm_output, latency = eval_vln(
                    frame,
                    server_url=args.server_url,
                    instruction=args.instruction,
                    policy_init=policy_init,
                    timeout_sec=args.timeout,
                )
                policy_init = False
                last_actions = actions
                last_status = status
                last_analysis = analysis
                last_llm_output = llm_output
                last_latency = latency
                last_call_ts = now

                print(
                    f"[frame {frame_id}] "
                    f"Instruction={truncate_text(args.instruction, 100)} | "
                    f"Actions={actions} ({actions_to_text(actions)}) | "
                    f"Status={truncate_text(status, 60)} | "
                    f"Analysis={truncate_text(analysis, 120)} | "
                    f"LLM output={truncate_text(llm_output, 140)} | "
                    f"Latency={latency:.3f}s"
                )
            except requests.exceptions.ConnectionError as e:
                print(f"[frame {frame_id}] connection failed: {e}")
            except requests.exceptions.Timeout as e:
                print(f"[frame {frame_id}] timeout after {args.timeout:.1f}s: {e}")
            except requests.exceptions.RequestException as e:
                print(f"[frame {frame_id}] request failed: {e}")
            except ValueError as e:
                print(f"[frame {frame_id}] response format invalid: {e}")
            except Exception as e:
                print(f"[frame {frame_id}] unexpected error: {e}")

        vis = draw_overlay(
            frame,
            [
                f"Instruction: {truncate_text(args.instruction, 70)}",
                f"Actions: {actions_to_text(last_actions)}",
                f"Status: {truncate_text(last_status, 85)}",
                f"Analysis: {truncate_text(last_analysis, 85)}",
                f"LLM output: {truncate_text(last_llm_output, 85)}",
                f"Latency: {last_latency:.3f}s",
                f"Frame: {frame_id}",
                f"Server: {args.server_url}",
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

