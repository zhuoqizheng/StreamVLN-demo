import threading
import time
import numpy as np
import math
import os
import sys

# 加入d1_vln_client.py路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from d1_vln_client import pose_matrix, PID_controller, control_thread, planning_thread

# 模拟D1VlnManager，只用虚拟数据，不用ROS
class FakeManager:
    def __init__(self):
        self.rgb_image = None
        self.homo_goal = None
        self.homo_odom = None
        self.vel = None
        self.request_cnt = 0
        self.should_plan = False
        self.instruction = "Walk forward and turn right."
        self.step_forward_m = 0.25
        self.turn_angle_rad = math.radians(15.0)
    def move(self, vx, vy, vyaw):
        print(f"[Fake Move] vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}")
    def trigger_replan(self):
        self.should_plan = True
    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_goal
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += self.step_forward_m * np.cos(yaw)
                homo_goal[1, 3] += self.step_forward_m * np.sin(yaw)
            elif each_action == 2:
                angle = self.turn_angle_rad
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle),  math.cos(angle), 0],
                    [0,                0,               1]
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -self.turn_angle_rad
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle),  math.cos(angle), 0],
                    [0,                0,               1]
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal

def fake_rgb_image():
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

def fake_odom(x=0.0, y=0.0, yaw=0.0, vx=0.0, wz=0.0):
    homo_odom = pose_matrix(x, y, yaw)
    vel = np.array([vx, wz], dtype=np.float64)
    return homo_odom, vel

def main():
    import rclpy
    global manager
    print("Fake test mode: initializing rclpy and FakeManager...")
    rclpy.init()
    manager = FakeManager()
    # 初始化虚拟数据
    manager.rgb_image = fake_rgb_image()
    manager.homo_odom, manager.vel = fake_odom(x=1.0, y=2.0, yaw=np.pi/4, vx=0.1, wz=0.05)
    manager.homo_goal, _ = fake_odom(x=2.0, y=3.0, yaw=np.pi/2)
    manager.should_plan = True
    t1 = threading.Thread(target=control_thread, daemon=True)
    t2 = threading.Thread(target=planning_thread, daemon=True)
    t1.start()
    t2.start()
    try:
        # 用rclpy.spin模拟主循环
        for _ in range(30):
            time.sleep(0.2)
        print("Fake test finished.")
    except KeyboardInterrupt:
        print("Test interrupted by user.")
    finally:
        rclpy.shutdown()
        print("Test finished.")

if __name__ == "__main__":
    main()
