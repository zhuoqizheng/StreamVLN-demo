import os
import csv
import threading
import time
import math
import json
import io
from datetime import datetime

# Third-party
import numpy as np
import requests
import PIL.Image as PIL_Image

# ROS
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TwistStamped
from cv_bridge import CvBridge

# Project modules
from utils import ReadWriteLock
from pid_controller_v2 import PID_controller   # ← 使用上面优化后的 pid_controller.py

# ================= 可调参数区域 =================

# VLN模型服务地址
VLN_SERVER_URL = os.environ.get('STREAMVLN_SERVER_URL', 'http://192.168.51.172:5801/eval_vln')
# 指令文本
DEFAULT_INSTRUCTION = os.environ.get('STREAMVLN_INSTRUCTION', 'go straight along the office corridor. when you see the paper box, turn left , then go fowards, stop in front of the door')


# PID参数（已优化：低Kp + 高Kd + 死区 + 平滑）
PID_KP_TRANS = 0.45
PID_KD_TRANS = 0.80
PID_KP_YAW   = 0.60
PID_KD_YAW   = 1.2
PID_MAX_V    = 0.40
PID_MAX_W    = 1.5

# 步进距离和转角（保持不变）
STEP_FORWARD_M = 0.25
TURN_ANGLE_DEG = 15.0
TURN_ANGLE_RAD = math.radians(TURN_ANGLE_DEG)

# ROS topic
RGB_TOPIC = '/odin1/image/undistorted'
ODOM_TOPIC = '/odin1/odometry'
CMD_VEL_TOPIC = '/cmd_vel'

# ================= 以下代码完全不变 =================
class ExperimentLogger:
    def __init__(self, log_dir='experiment_logs', log_name=None, instruction_getter=None):
        os.makedirs(log_dir, exist_ok=True)
        if log_name is None:
            log_name = datetime.now().strftime('exp_%Y%m%d_%H%M%S.csv')
        self.log_path = os.path.join(log_dir, log_name)
        self.lock = threading.Lock()
        self.header_written = False
        self.metadata_written = False
        self.instruction_getter = instruction_getter

    def _write_metadata_if_needed(self, file_obj):
        if self.metadata_written:
            return
        instruction = self.instruction_getter() if self.instruction_getter is not None else None
        if instruction:
            file_obj.write(f'# instruction: {instruction}\n')
        self.metadata_written = True

    def log(self, data_dict):
        with self.lock:
            write_header = not os.path.exists(self.log_path) or not self.header_written
            with open(self.log_path, 'a', newline='') as f:
                self._write_metadata_if_needed(f)
                writer = csv.DictWriter(f, fieldnames=data_dict.keys())
                if write_header:
                    writer.writeheader()
                    self.header_written = True
                writer.writerow(data_dict)

exp_logger = ExperimentLogger(
    instruction_getter=lambda: manager.instruction if manager is not None else DEFAULT_INSTRUCTION
)

# global variable
policy_init = True
pid = PID_controller(Kp_trans=PID_KP_TRANS, Kd_trans=PID_KD_TRANS, 
                     Kp_yaw=PID_KP_YAW, Kd_yaw=PID_KD_YAW, 
                     max_v=PID_MAX_V, max_w=PID_MAX_W)
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
    global policy_init
    if url is None:
        url = VLN_SERVER_URL
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
    elapsed = time.time() - start
    print(f"total time(delay + policy): {elapsed}")
    print(response.text)
    action = json.loads(response.text)['action']
    exp_logger.log({
        'event': 'eval_vln',
        'timestamp': time.time(),
        'elapsed': elapsed,
        'actions': str(action),
        'response': response.text
    })
    return action

def control_thread():
    GREEN = '\033[92m'
    RESET = '\033[0m'
    while True:
        odom_rw_lock.acquire_read()
        homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
        vel = manager.vel.copy() if manager.vel is not None else None
        homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None
        should_plan = manager.should_plan
        odom_rw_lock.release_read()

        if should_plan:
            manager.move(0.0, 0.0, 0.0)
            time.sleep(0.05)
            continue
        
        e_p, e_r = 0.0, 0.0
        extract_pose = None
        if homo_odom is not None and vel is not None and homo_goal is not None:
            v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)

            if abs(e_p) < 0.1 and abs(e_r) < 0.1:
                manager.move(0.0, 0.0, 0.0)
                pid.reset(odom=homo_odom)
                manager.trigger_replan()
                print(f'{GREEN}replan triggered{RESET}')

                def extract_pose(mat):
                    x = float(mat[0, 3])
                    y = float(mat[1, 3])
                    yaw = math.atan2(mat[1, 0], mat[0, 0])
                    return x, y, yaw

                odom_x, odom_y, odom_yaw = extract_pose(homo_odom)
                goal_x, goal_y, goal_yaw = extract_pose(homo_goal)
                exp_logger.log({
                    'event': 'replan',
                    'timestamp': time.time(),
                    'e_p': e_p,
                    'e_r': e_r,
                    'odom_x': odom_x,
                    'odom_y': odom_y,
                    'odom_yaw': odom_yaw,
                    'goal_x': goal_x,
                    'goal_y': goal_y,
                    'goal_yaw': goal_yaw
                })
                time.sleep(0.05)
                continue
            
            # 保留你原有的小死区处理（作为双保险）
            if abs(w) < 0.02:
                w = 0.0
            w = max(min(w, 3.0), -3.0)
            
            manager.move(v, 0, w)
            # print(f'{GREEN}command: v={v:.3f}, w={w:.3f}, e_p={e_p:.3f}, e_r={e_r:.3f}{RESET}')
            
            # 日志记录（保持不变）
            def extract_pose(mat):
                x = float(mat[0, 3])
                y = float(mat[1, 3])
                yaw = math.atan2(mat[1, 0], mat[0, 0])
                return x, y, yaw
            odom_x, odom_y, odom_yaw = extract_pose(homo_odom)
            goal_x, goal_y, goal_yaw = extract_pose(homo_goal)
            exp_logger.log({
                'event': 'control',
                'timestamp': time.time(),
                'v': v,
                'w': w,
                'e_p': e_p,
                'e_r': e_r,
                'odom_x': odom_x,
                'odom_y': odom_y,
                'odom_yaw': odom_yaw,
                'goal_x': goal_x,
                'goal_y': goal_y,
                'goal_yaw': goal_yaw
            })
        time.sleep(0.1)

def planning_thread():
    # 根据当前RGB图像和指令，调用VLN模型进行路径规划
    # 此处调用了VLN
    CYAN = '\033[96m'
    RESET = '\033[0m'
    while True:
        if not manager.should_plan:
            time.sleep(0.05)
            continue
        # print(f"{CYAN}planning_thread running{RESET}")
        rgb_rw_lock.acquire_read()
        rgb_image = manager.rgb_image.copy() if manager.rgb_image is not None else None
        rgb_rw_lock.release_read()
        if rgb_image is None:
            time.sleep(0.1)
            continue

        actions = eval_vln(rgb_image, manager.instruction)
        print(f"{CYAN}actions: {actions}{RESET}")
        exp_logger.log({
            'event': 'planning',
            'timestamp': time.time(),
            'actions': str(actions)
        })
        odom_rw_lock.acquire_write()
        if manager.homo_odom is not None:
            manager.homo_goal = manager.homo_odom.copy()
        manager.should_plan = False
        manager.request_cnt += 1
        manager.incremental_change_goal(actions)
        pid.reset(target=manager.homo_goal)
        odom_rw_lock.release_write()
        time.sleep(0.1)

class D1VlnManager(object):
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.homo_goal = None
        self.homo_odom = None
        self.vel = None
        self.request_cnt = 0
        self.should_plan = False
        self.instruction = DEFAULT_INSTRUCTION
        self.step_forward_m = STEP_FORWARD_M
        self.turn_angle_rad = TURN_ANGLE_RAD
        # 机器狗ROS接口
        self.rgb_sub = rospy.Subscriber(RGB_TOPIC, Image, self.rgb_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_callback, queue_size=10)
        self.cmd_pub = rospy.Publisher(CMD_VEL_TOPIC, TwistStamped, queue_size=10)
        rospy.loginfo(f"rgb_topic={RGB_TOPIC}, odom_topic={ODOM_TOPIC}, cmd_vel_topic={CMD_VEL_TOPIC}")

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
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        # msg.header.frame_id = "base_link"  # 可根据需要修改
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.angular.z = float(vyaw)
        self.cmd_pub.publish(msg)
   

if __name__ == '__main__':
    rospy.init_node('d1_vln_client')
    manager = D1VlnManager()
    t1 = threading.Thread(target=control_thread, daemon=True)
    t2 = threading.Thread(target=planning_thread, daemon=True)
    t1.start()
    t2.start()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        pass