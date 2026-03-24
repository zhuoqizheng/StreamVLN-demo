import argparse
import numpy as np
import json
import time
import torch
import sys
import os
import transformers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from streamvln.streamvln_agent import VLNEvaluator
from model.stream_video_vln import StreamVLNForCausalLM

app = Flask(__name__)
action_seq = np.zeros(4)
idx = 0
terminate = False
total_generate_time = 0.0
start_time = time.time()
output_dir = '/tf/StreamVLN/data_image'


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _to_action_list(action_arr):
    if action_arr is None:
        return [0]
    if isinstance(action_arr, np.ndarray):
        return [int(x) for x in action_arr.tolist()]
    return [int(x) for x in action_arr]


def annotate_image(idx, image, start_time, total_generate_time, llm_output, output_dir):
    image = Image.fromarray(image)#.save(f'rgb_{idx}.png')
    draw = ImageDraw.Draw(image)
    font_size = 20
    font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    text_content = []
    text_content.append(f"Frame    Id  : {idx}")
    text_content.append(f"Running  time: {time.time() - start_time:.2f} s")
    text_content.append(f"Generate time: {total_generate_time:.2f} s")
    text_content.append(f"Actions      : {llm_output}" )
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

    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

    text_color = 'white'
    y_position = box_y + padding
    
    for line in text_content:
        draw.text((box_x + padding, y_position), line, fill=text_color, font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        text_height = 26
        y_position += text_height

    image.save(f'{output_dir}/rgb_{idx}_annotated.png')

@app.route("/eval_vln",methods=['POST'])
def eval_vln():
    global action_seq, idx, terminate, total_generate_time, output_dir, start_time

    image_file = request.files.get('image') or request.files.get('file')
    if image_file is None:
        return jsonify({'error': 'missing multipart image field: image or file'}), 400

    data = {}
    json_data = request.form.get('json')
    if json_data:
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            return jsonify({'error': 'invalid json field in multipart form'}), 400

    # Compatible with multiple clients: JSON payload or plain multipart form keys.
    policy_init = _parse_bool(data.get('reset', request.form.get('reset', False)))
    instruction = (
        (data.get('instruction') if isinstance(data, dict) else None)
        or request.form.get('instruction')
        or request.form.get('instruction_text')
        or "Walk forward and immediately stop when you exit the room."
    )
    
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)[...,::-1]

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    if policy_init:
        start_time = time.time()
        total_generate_time = 0.0
        terminate = False
        idx = 0
        output_dir = 'runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        evaluator.reset_memory()
    
    idx += 1
    
    if terminate:
        print("!!!!!!!!!!!!!!!!!task finish!!!!!!!!!!!!!!!!!!!!!")
        return jsonify({'action': [0]})

    llm_output = ""
    for i in range(4):
        t1 = time.time()
        depth = np.zeros((image.shape[0], image.shape[1], 1))
        return_action, generate_time, return_llm_output = evaluator.step(0,
                                        image,
                                        #depth,
                                        #camera_pose,
                                        instruction,
                                        run_model=(evaluator.step_id % 4 == 0))
        llm_output = return_llm_output if return_llm_output is not None else llm_output
        print(f"one evalute cost {time.time() - t1}")
        # total_generate_time += generate_time
        
        if generate_time > 0:
            total_generate_time = generate_time
        action_seq = action_seq if return_action is None else return_action
        if 0 in action_seq:
            terminate = True     
        evaluator.step_id += 1
        
    str_action = [str(i) for i in action_seq]
    str_action = ''.join(str_action)
    str_action = str_action.replace('1', '↑')  # 前箭头
    str_action = str_action.replace('2', '←')  # 左箭头
    str_action = str_action.replace('3', '→')  # 右箭头
    str_action = str_action.replace('0', 'STOP')  # 停止
    if idx > 1 and total_generate_time > 0.5:
        total_generate_time -= 0.3

    annotate_image(idx, image, start_time, total_generate_time, str_action, output_dir)

    if len(action_seq) == 0:
        print("!!!!!!!!!!!!!!!!!task finish!!!!!!!!!!!!!!!!!!!!!")
        return jsonify({'action': [0]})

    return jsonify({
        'action': _to_action_list(action_seq),
        'instruction': instruction,
        'frame_id': idx,
        'terminate': bool(terminate),
    })


@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({
        'ok': True,
        'step_id': int(getattr(evaluator, 'step_id', -1)),
        'terminate': bool(terminate),
    })
    
if __name__ == '__main__':
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/tf/StreamVLN/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='device to use for testing (Jetson Orin: cuda:0)')
    
    args = parser.parse_args()
    if isinstance(args.device, str) and args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"[WARN] Requested device '{args.device}' but CUDA is unavailable. Falling back to CPU.")
        args.device = 'cpu'

    model_dtype = torch.bfloat16 if args.device.startswith('cuda') else torch.float32

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
        "rgb_height" : 1.25, 
        "camera_intrinsic" : np.array([[192.        ,   0.        , 191.42857143,   0.        ],
            [  0.        , 192.        , 191.42857143,   0.        ],
            [  0.        ,   0.        ,   1.        ,   0.        ],
            [  0.        ,   0.        ,   0.        ,   1.        ]]),
    }
    
    evaluator = VLNEvaluator(
        vln_sensor_config,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )
    
    
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=True)
    app.run(host='0.0.0.0', port=
            5801)
