import io
import json
import math
import os
import platform
import sys
import threading
import time

import PIL.Image as PIL_Image
import numpy as np
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from pid_controller import PID_controller
from utils import ReadWriteLock


policy_init = True
manager = None
pid = PID_controller(
    Kp_trans=3.0,
    Kd_trans=0.5,
    Kp_yaw=3.0,
    Kd_yaw=0.5,
    max_v=1.0,
    max_w=1.2,
)

rgb_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()


def eval_vln(image, instruction=None, url=None):
    global policy_init

    if url is None:
        url = os.environ.get("STREAMVLN_SERVER_URL", "http://localhost:5801/eval_vln")

    image = PIL_Image.fromarray(image)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="jpeg")
    image_bytes.seek(0)

    data = {"reset": policy_init}
    instruction_text = instruction or os.environ.get("STREAMVLN_INSTRUCTION")
    if instruction_text:
        data["instruction"] = instruction_text
    json_data = json.dumps(data)
    policy_init = False

    timeout_s = float(os.environ.get("STREAMVLN_TIMEOUT", "150"))
    max_retry = int(os.environ.get("STREAMVLN_RETRY", "2"))

    last_err = None
    for _ in range(max_retry + 1):
        try:
            files = {"image": ("rgb_image", image_bytes, "image/jpg")}
            image_bytes.seek(0)
            start = time.time()
            response = requests.post(url, files=files, data={"json": json_data}, timeout=timeout_s)
            print(f"total time(delay + policy): {time.time() - start}")
            print(response.text)
            response.raise_for_status()

            payload = response.json()
            action = payload.get("action", payload.get("actions", [0]))
            return [int(a) for a in action]
        except Exception as exc:
            last_err = exc
            time.sleep(0.2)

    raise RuntimeError(f"eval_vln request failed after retries: {last_err}")


def _normalize_arch():
    return platform.machine().replace("amd64", "x86_64").replace("arm64", "aarch64")


def _load_d1_sdk_module():
    model = os.environ.get("D1_MODEL", "zsl-1").strip().lower()
    if model not in ("zsl-1", "zsl-1w"):
        raise ValueError("D1_MODEL must be one of: zsl-1, zsl-1w")

    module_name = "mc_sdk_zsl_1_py" if model == "zsl-1" else "mc_sdk_zsl_1w_py"

    explicit_lib_path = os.environ.get("D1_SDK_LIB_PATH")
    sdk_repo_root = os.environ.get("D1_SDK_ROOT")
    if explicit_lib_path:
        sys.path.insert(0, explicit_lib_path)
    elif sdk_repo_root:
        arch = _normalize_arch()
        lib_path = os.path.abspath(os.path.join(sdk_repo_root, "lib", model, arch))
        sys.path.insert(0, lib_path)

    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            "Cannot import D1 SDK Python module. Set D1_SDK_ROOT to the agibot_D1_Edu-Ultra "
            "repo path, or set D1_SDK_LIB_PATH to the folder containing "
            f"{module_name}.so"
        ) from exc


def control_thread(stop_event):
    while not stop_event.is_set():
        if manager is None:
            time.sleep(0.1)
            continue

        manager.refresh_state()

        odom_rw_lock.acquire_read()
        homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
        vel = manager.vel.copy() if manager.vel is not None else None
        homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None
        odom_rw_lock.release_read()

        e_p, e_r = 0.0, 0.0
        if homo_odom is not None and vel is not None and homo_goal is not None:
            v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
            manager.move(v, 0.0, w)

        if abs(e_p) < 0.1 and abs(e_r) < 0.1:
            manager.trigger_replan()

        time.sleep(0.1)


def planning_thread(stop_event):
    while not stop_event.is_set():
        if manager is None:
            time.sleep(0.1)
            continue

        if not manager.should_plan:
            time.sleep(0.05)
            continue

        print("planning_thread running")

        rgb_rw_lock.acquire_read()
        rgb_image = manager.rgb_image
        rgb_rw_lock.release_read()

        if rgb_image is None:
            time.sleep(0.1)
            continue

        actions = eval_vln(rgb_image)

        odom_rw_lock.acquire_write()
        manager.should_plan = False
        manager.request_cnt += 1
        manager.incremental_change_goal(actions)
        odom_rw_lock.release_write()

        time.sleep(0.1)


class D1VlnManager(Node):
    def __init__(self):
        super().__init__("d1_vln_manager")

        self.cv_bridge = CvBridge()
        self.rgb_image = None

        self.homo_goal = None
        self.homo_odom = None
        self.vel = None

        self.request_cnt = 0
        self.should_plan = False
        self._last_replan_t = 0.0

        rgb_topic = os.environ.get("D1_RGB_TOPIC", "/camera/camera/color/image_raw")
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 1)

        sdk_module = _load_d1_sdk_module()
        self.app = sdk_module.HighLevel()

        local_ip = os.environ.get("D1_LOCAL_IP", "127.0.0.1")
        local_port = int(os.environ.get("D1_LOCAL_PORT", "43988"))
        robot_ip = os.environ.get("D1_ROBOT_IP", "192.168.234.1")

        self.app.initRobot(local_ip, local_port, robot_ip)
        time.sleep(0.5)

        if hasattr(self.app, "checkConnection") and not self.app.checkConnection():
            raise RuntimeError("D1 SDK is not connected. Please check IP/port/network config.")

        ret = self.app.standUp()
        self.get_logger().info(f"standUp return: {ret}")
        time.sleep(3.0)

        self.refresh_state()
        self.trigger_replan()

    def rgb_callback(self, msg):
        rgb_rw_lock.acquire_write()
        raw_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")[:, :, :]
        self.rgb_image = raw_image
        rgb_rw_lock.release_write()

    def refresh_state(self):
        pos = self.app.getPosition() if hasattr(self.app, "getPosition") else [0.0, 0.0, 0.0]
        rpy = self.app.getRPY() if hasattr(self.app, "getRPY") else [0.0, 0.0, 0.0]

        body_vel = self.app.getBodyVelocity() if hasattr(self.app, "getBodyVelocity") else [0.0, 0.0, 0.0]
        body_gyro = self.app.getBodyGyro() if hasattr(self.app, "getBodyGyro") else [0.0, 0.0, 0.0]

        yaw = float(rpy[2])
        x = float(pos[0])
        y = float(pos[1])

        vx = float(body_vel[0])
        wz = float(body_gyro[2])

        odom_rw_lock.acquire_write()
        homo_odom = np.eye(4)
        homo_odom[0, 0] = math.cos(yaw)
        homo_odom[0, 1] = -math.sin(yaw)
        homo_odom[1, 0] = math.sin(yaw)
        homo_odom[1, 1] = math.cos(yaw)
        homo_odom[0, 3] = x
        homo_odom[1, 3] = y

        self.homo_odom = homo_odom
        self.vel = [vx, wz]

        if self.homo_goal is None:
            self.homo_goal = self.homo_odom.copy()
        odom_rw_lock.release_write()

    def trigger_replan(self):
        now = time.time()
        if now - self._last_replan_t > 0.5:
            self.should_plan = True
            self._last_replan_t = now

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")

        homo_goal = self.homo_goal
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * math.cos(yaw)
                homo_goal[1, 3] += 0.25 * math.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15.0)
                rotation_matrix = np.array(
                    [
                        [math.cos(angle), -math.sin(angle), 0.0],
                        [math.sin(angle), math.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [
                        [math.cos(angle), -math.sin(angle), 0.0],
                        [math.sin(angle), math.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])

        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        del vy
        self.app.move(float(vx), 0.0, float(vyaw))

    def shutdown(self):
        try:
            self.app.move(0.0, 0.0, 0.0)
            if os.environ.get("D1_LIE_DOWN_ON_EXIT", "0") in ("1", "true", "True"):
                self.app.lieDown()
        except Exception as exc:
            self.get_logger().warning(f"shutdown command failed: {exc}")


if __name__ == "__main__":
    stop_event = threading.Event()
    control_thread_instance = threading.Thread(target=control_thread, args=(stop_event,), daemon=True)
    planning_thread_instance = threading.Thread(target=planning_thread, args=(stop_event,), daemon=True)

    rclpy.init()

    try:
        manager = D1VlnManager()

        control_thread_instance.start()
        planning_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if manager is not None:
            manager.shutdown()
            manager.destroy_node()
        rclpy.shutdown()
