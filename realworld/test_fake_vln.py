import numpy as np
from PIL import Image
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 假设eval_vln函数和pose_matrix函数与d1_vln_client.py一致
from d1_vln_client import eval_vln, pose_matrix, PID_controller

def generate_fake_image(width=224, height=224):
    # 生成一张随机RGB图像
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

def generate_fake_odom(x=0.0, y=0.0, yaw=0.0, vx=0.0, wz=0.0):
    # 生成虚拟odom数据
    homo_odom = pose_matrix(x, y, yaw)
    vel = np.array([vx, wz], dtype=np.float64)
    return homo_odom, vel

def main():
    # 生成虚拟图像
    fake_img = generate_fake_image()
    # 生成虚拟odom
    homo_odom, vel = generate_fake_odom(x=1.0, y=2.0, yaw=np.pi/4, vx=0.1, wz=0.05)
    # 指令
    instruction = "Walk forward and turn right."
    # 调用eval_vln
    print("Testing eval_vln with fake image...")
    actions = eval_vln(fake_img, instruction)
    print(f"Returned actions: {actions}")
    # PID控制器测试
    pid = PID_controller(Kp_trans=3.0, Kd_trans=0.5, Kp_yaw=3.0, Kd_yaw=0.5, max_v=1.0, max_w=1.2)
    # 目标点
    homo_goal, _ = generate_fake_odom(x=2.0, y=3.0, yaw=np.pi/2)
    v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
    print(f"PID output: v={v}, w={w}, e_p={e_p}, e_r={e_r}")

if __name__ == "__main__":
    main()
