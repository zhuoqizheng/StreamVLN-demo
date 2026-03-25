import io
import json
import math
import os
import threading
import time

import PIL.Image as PIL_Image
import numpy as np
import requests
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

from pid_controller import *
from utils import ReadWriteLock

# global variable
policy_init = True
pid = PID_controller(Kp_trans=3.0, Kd_trans=0.5, Kp_yaw=3.0, Kd_yaw=0.5, max_v=1.0, max_w=1.2)
manager = None

rgb_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def pose_matrix(x, y, yaw):
    t = np.eye(4, dtype=np.float64)
    t[0, 0] = math.cos(yaw)
    t[0, 1] = -math.sin(yaw)
    t[1, 0] = math.sin(yaw)
    t[1, 1] = math.cos(yaw)
    t[0, 3] = x
    t[1, 3] = y
    return t

def eval_vln(image, instruction=None, url=None):
    # 调用VLN模型接口，返回动作列表
    global policy_init
    if url is None:
        url = os.environ.get('STREAMVLN_SERVER_URL', 'http://localhost:5801/eval_vln')
    image = PIL_Image.fromarray(image)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='jpeg')
    image_bytes.seek(0)
    data = {"reset": policy_init}
    instruction_text = instruction or os.environ.get('STREAMVLN_INSTRUCTION')
    if instruction_text:
        data['instruction'] = instruction_text
    json_data = json.dumps(data)
    policy_init = False
    files = {'image': ('rgb_image', image_bytes, 'image/jpg')}
    start = time.time()
    response = requests.post(url, files=files, data={'json': json_data}, timeout=150)
    print(f"total time(delay + policy): {time.time() - start}")
    print(response.text)
    action = json.loads(response.text)['action']
    return action

def control_thread():
    # 根据当前odom和goal，计算控制指令并发布
    while True:
        odom_rw_lock.acquire_read()
        homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
        vel = manager.vel.copy() if manager.vel is not None else None
        homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None
        odom_rw_lock.release_read()
        e_p, e_r = 0.0, 0.0
        if homo_odom is not None and vel is not None and homo_goal is not None:
            v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
            manager.move(v, 0, w)
        if abs(e_p) < 0.1 and abs(e_r) < 0.1:
            manager.trigger_replan()
        time.sleep(0.1)

def planning_thread():
    # 根据当前RGB图像和指令，调用VLN模型进行路径规划
    # 此处调用了VLN
    while True:
        if not manager.should_plan:
            time.sleep(0.05)
            continue
        print(f"planning_thread running")
        rgb_rw_lock.acquire_read()
        rgb_image = manager.rgb_image.copy() if manager.rgb_image is not None else None
        rgb_rw_lock.release_read()
        if rgb_image is None:
            time.sleep(0.1)
            continue
        actions = eval_vln(rgb_image, manager.instruction)
        print(f"actions: {actions}")
        odom_rw_lock.acquire_write()
        manager.should_plan = False
        manager.request_cnt += 1
        manager.incremental_change_goal(actions)
        odom_rw_lock.release_write()
        time.sleep(0.1)

class D1VlnManager(Node):
    def __init__(self):
        super().__init__('odin_vln_client')
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.homo_goal = None
        self.homo_odom = None
        self.vel = None
        self.request_cnt = 0
        self.should_plan = False
        self.instruction = os.environ.get(
            "STREAMVLN_INSTRUCTION",
            "Walk forward and immediately stop when you exit the room.",
        )
        self.step_forward_m = float(os.environ.get("ODIN_STEP_FORWARD_M", "0.25"))
        self.turn_angle_rad = math.radians(float(os.environ.get("ODIN_TURN_ANGLE_DEG", "15.0")))
        # 机器狗ROS接口
        rgb_topic = os.environ.get("ODIN_RGB_TOPIC", os.environ.get("D1_RGB_TOPIC", "/odin1/image/undistorted"))
        odom_topic = os.environ.get("ODIN_ODOM_TOPIC", "/odin1/odometry_highfreq")
        cmd_vel_topic = os.environ.get("ODIN_CMD_VEL_TOPIC", "/cmd_vel")
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.get_logger().info(
            f"rgb_topic={rgb_topic}, odom_topic={odom_topic}, cmd_vel_topic={cmd_vel_topic}"
        )

    def rgb_callback(self, msg):
        rgb_rw_lock.acquire_write()
        raw_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')[:, :, :]
        self.rgb_image = raw_image
        rgb_rw_lock.release_write()

    def odom_callback(self, msg):
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        vx = float(msg.twist.twist.linear.x)
        wz = float(msg.twist.twist.angular.z)
        odom_rw_lock.acquire_write()
        self.homo_odom = pose_matrix(x, y, yaw)
        self.vel = np.array([vx, wz], dtype=np.float64)
        if self.homo_goal is None:
            self.homo_goal = self.homo_odom.copy()
            self.trigger_replan()
        odom_rw_lock.release_write()

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

    def move(self, vx, vy, vyaw):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(vyaw)
        self.cmd_pub.publish(msg)

if __name__ == '__main__':
    rclpy.init()
    global manager
    manager = D1VlnManager()
    t1 = threading.Thread(target=control_thread, daemon=True)
    t2 = threading.Thread(target=planning_thread, daemon=True)
    t1.start()
    t2.start()
    try:
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        rclpy.shutdown()
