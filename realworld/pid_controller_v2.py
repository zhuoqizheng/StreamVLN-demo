import math
import numpy as np

class PID_controller:
    def __init__(self, 
                 Kp_trans=0.35, 
                 Kd_trans=0.80, 
                 Kp_yaw=0.60, 
                 Kd_yaw=0.85, 
                 max_v=0.25, 
                 max_w=0.60):
        
        self.Kp_trans = Kp_trans
        self.Kd_trans = Kd_trans
        self.Kp_yaw = Kp_yaw
        self.Kd_yaw = Kd_yaw
        self.max_v = max_v
        self.max_w = max_w
        
        # 新增优化参数（针对10Hz + 小误差震荡）
        self.alpha_pos = 0.14          # 指数平滑系数（可调）
        self.deadzone_trans = 0.04     # 位置误差死区（4cm以内v=0）
        self.deadzone_yaw = 0.08       # 旋转误差死区（约4.5°）
        self.max_delta_v = 0.18        # 每周期最大速度变化（加速度限制）
        self.max_delta_w = 0.25        # 每周期最大角速度变化
        
        self.smoothed_target = None
        self.last_v = 0.0
        self.last_w = 0.0

    def reset(self, odom=None, target=None):
        if target is not None:
            self.smoothed_target = np.array([target[0, 3], target[1, 3]], dtype=np.float64)
        elif odom is not None:
            self.smoothed_target = np.array([odom[0, 3], odom[1, 3]], dtype=np.float64)
        else:
            self.smoothed_target = None
        self.last_v = 0.0
        self.last_w = 0.0

    def solve(self, odom, target, vel=np.zeros(2)):
        if self.smoothed_target is None:
            self.smoothed_target = np.array([odom[0, 3], odom[1, 3]])
        
        target_pos = np.array([target[0, 3], target[1, 3]])
        
        # 指数平滑参考位置（解决高层目标突变）
        self.smoothed_target = (1 - self.alpha_pos) * self.smoothed_target + self.alpha_pos * target_pos
        
        # 构造平滑后的target（只平滑位置，朝向保持原目标）
        smoothed_target = target.copy()
        smoothed_target[0, 3] = self.smoothed_target[0]
        smoothed_target[1, 3] = self.smoothed_target[1]
        
        translation_error, yaw_error = self.calculate_errors(odom, smoothed_target)
        
        # 死区（解决误差小时震荡的核心）
        if abs(translation_error) < self.deadzone_trans:
            translation_error = 0.0
        if abs(yaw_error) < self.deadzone_yaw:
            yaw_error = 0.0
        
        v, w = self.pd_step(translation_error, yaw_error, vel[0], vel[1])
        return v, w, translation_error, yaw_error

    def pd_step(self, translation_error, yaw_error, linear_vel, angular_vel):
        linear_velocity = self.Kp_trans * translation_error - self.Kd_trans * linear_vel
        angular_velocity = self.Kp_yaw * yaw_error - self.Kd_yaw * angular_vel

        # 速率限制（防止前后移动）
        linear_velocity = max(self.last_v - self.max_delta_v,
                              min(self.last_v + self.max_delta_v, linear_velocity))
        self.last_v = linear_velocity

        angular_velocity = max(self.last_w - self.max_delta_w,
                               min(self.last_w + self.max_delta_w, angular_velocity))
        self.last_w = angular_velocity

        # 最终限幅
        linear_velocity = max(-self.max_v, min(self.max_v, linear_velocity))
        angular_velocity = max(-self.max_w, min(self.max_w, angular_velocity))

        # 彻底禁止微小负速度（避免前后抖动）
        if -0.05 < linear_velocity < 0:
            linear_velocity = 0.0

        return linear_velocity, angular_velocity

    def calculate_errors(self, odom, target):
        dx = target[0, 3] - odom[0, 3]
        dy = target[1, 3] - odom[1, 3]

        odom_yaw = math.atan2(odom[1, 0], odom[0, 0])
        target_yaw = math.atan2(target[1, 0], target[0, 0])

        translation_error = dx * np.cos(odom_yaw) + dy * np.sin(odom_yaw)

        yaw_error = target_yaw - odom_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        return translation_error, yaw_error