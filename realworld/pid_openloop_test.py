import rospy
from geometry_msgs.msg import Twist
import numpy as np
import math
import time

# --- PID Controller ---
class PID_controller:
    def __init__(self, Kp_trans=3.0, Kd_trans=0.5, Kp_yaw=3.0, Kd_yaw=0.5, max_v=1.0, max_w=1.5):
        self.Kp_trans = Kp_trans
        self.Kd_trans = Kd_trans
        self.Kp_yaw = Kp_yaw
        self.Kd_yaw = Kd_yaw
        self.max_v = max_v
        self.max_w = max_w
        self.last_e_p = 0.0
        self.last_e_r = 0.0
    def solve(self, homo_odom, homo_goal, vel):
        dx = homo_goal[0, 3] - homo_odom[0, 3]
        dy = homo_goal[1, 3] - homo_odom[1, 3]
        e_p = math.hypot(dx, dy)
        yaw = math.atan2(homo_odom[1, 0], homo_odom[0, 0])
        goal_yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
        e_r = goal_yaw - yaw
        e_r = (e_r + np.pi) % (2 * np.pi) - np.pi
        v = self.Kp_trans * e_p + self.Kd_trans * (e_p - self.last_e_p)
        w = self.Kp_yaw * e_r + self.Kd_yaw * (e_r - self.last_e_r)
        v = np.clip(v, -self.max_v, self.max_v)
        w = np.clip(w, -self.max_w, self.max_w)
        self.last_e_p = e_p
        self.last_e_r = e_r
        return v, w, e_p, e_r

# --- Goal Update ---
def incremental_change_goal(homo_goal, actions, step_forward_m=0.25, turn_angle_deg=15.0):
    turn_angle_rad = math.radians(turn_angle_deg)
    homo_goal = np.array(homo_goal, copy=True)
    for each_action in actions:
        if each_action == 0:
            pass
        elif each_action == 1:
            yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
            homo_goal[0, 3] += step_forward_m * np.cos(yaw)
            homo_goal[1, 3] += step_forward_m * np.sin(yaw)
        elif each_action == 2:
            angle = turn_angle_rad
            rotation_matrix = np.array([
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle),  math.cos(angle), 0],
                [0,                0,               1]
            ])
            homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        elif each_action == 3:
            angle = -turn_angle_rad
            rotation_matrix = np.array([
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle),  math.cos(angle), 0],
                [0,                0,               1]
            ])
            homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
    return homo_goal

# --- Main Openloop Test ---
if __name__ == "__main__":
    rospy.init_node('pid_openloop_test')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    print('输入指令: 0=停, 1=前进, 2=左转, 3=右转')
    # 初始状态
    homo_odom = np.eye(4)
    homo_goal = np.eye(4)
    vel = np.array([0.0, 0.0])
    pid = PID_controller()
    step_forward_m = 0.25
    turn_angle_deg = 15.0
    while not rospy.is_shutdown():
        a = input('Action序列(如1,1,2,1,3,0): ')
        try:
            actions = [int(x) for x in a.strip().split(',') if x.strip() in ['0','1','2','3']]
        except Exception:
            print('输入无效')
            continue
        if not actions:
            continue
        homo_goal = incremental_change_goal(homo_goal, actions, step_forward_m, turn_angle_deg)
        v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        pub.publish(msg)
        print(f'Published: v={v:.3f}, w={w:.3f}, e_p={e_p:.3f}, e_r={e_r:.3f}')
        time.sleep(0.2)
        # 停止
        stop_msg = Twist()
        for _ in range(10):
            pub.publish(stop_msg)
            time.sleep(0.05)
        print('Published stop cmd_vel (v=0, w=0)')
