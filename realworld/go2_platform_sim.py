import argparse
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from unitree_api.msg import Request
from unitree_go.msg import SportModeState


@dataclass
class ScenarioConfig:
    person_present: bool = True
    door_closed: bool = True
    trash_has_garbage: bool = False


class Go2PlatformSim(Node):
    def __init__(self, args):
        super().__init__("go2_platform_sim")
        self.args = args
        self.scenario = ScenarioConfig(
            person_present=args.person_present,
            door_closed=args.door_closed,
            trash_has_garbage=args.trash_has_garbage,
        )

        self.image_pub = self.create_publisher(Image, args.image_topic, 10)
        self.odom_pub = self.create_publisher(SportModeState, args.odom_topic, 10)
        self.cmd_sub = self.create_subscription(Request, args.cmd_topic, self.cmd_callback, 20)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx_cmd = 0.0
        self.wz_cmd = 0.0
        self.last_update = time.time()

        self.publish_timer = self.create_timer(1.0 / args.publish_hz, self.on_timer)
        self.get_logger().info(
            f"Go2 simulator started | image={args.image_topic} odom={args.odom_topic} cmd={args.cmd_topic}"
        )

    def cmd_callback(self, msg: Request):
        parameter = getattr(msg, "parameter", "")
        try:
            payload = json.loads(parameter) if parameter else {}
        except json.JSONDecodeError:
            self.get_logger().warning(f"Failed to decode command parameter: {parameter}")
            return

        self.vx_cmd = float(payload.get("x", 0.0))
        self.wz_cmd = float(payload.get("z", 0.0))
        self.get_logger().info(f"cmd vx={self.vx_cmd:.3f} wz={self.wz_cmd:.3f}")

    def on_timer(self):
        now = time.time()
        dt = max(1e-3, now - self.last_update)
        self.last_update = now

        self.yaw += self.wz_cmd * dt
        self.x += self.vx_cmd * math.cos(self.yaw) * dt
        self.y += self.vx_cmd * math.sin(self.yaw) * dt

        self.publish_odometry()
        self.publish_image()

    def publish_odometry(self):
        msg = SportModeState()
        msg.position = [float(self.x), float(self.y), 0.0]
        msg.velocity = [float(self.vx_cmd), 0.0, 0.0]
        msg.yaw_speed = float(self.wz_cmd)
        msg.imu_state.rpy = [0.0, 0.0, float(self.yaw)]
        self.odom_pub.publish(msg)

    def publish_image(self):
        image = self.render_camera_view()
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "sim_camera"
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = image.shape[1] * 3
        msg.data = image.tobytes()
        self.image_pub.publish(msg)

    def render_camera_view(self):
        height = self.args.image_height
        width = self.args.image_width
        canvas = np.full((height, width, 3), 235, dtype=np.uint8)

        horizon = height // 3
        canvas[:horizon] = np.array([210, 230, 255], dtype=np.uint8)
        canvas[horizon:] = np.array([225, 225, 225], dtype=np.uint8)
        canvas[horizon - 2:horizon + 2, :] = 180

        self.draw_center_guides(canvas)
        self.draw_status_bar(canvas)
        self.draw_world_objects(canvas)
        return canvas

    def draw_center_guides(self, canvas):
        h, w = canvas.shape[:2]
        canvas[:, w // 2 - 1:w // 2 + 1] = np.array([170, 170, 170], dtype=np.uint8)
        for depth_mark in range(1, 5):
            y = h - depth_mark * (h // 6)
            canvas[y:y + 1, :] = np.array([200, 200, 200], dtype=np.uint8)

    def draw_status_bar(self, canvas):
        text = [
            f"SIM GO2 | x={self.x:.2f} y={self.y:.2f} yaw={math.degrees(self.yaw):.1f}",
            f"person={self.scenario.person_present} door_closed={self.scenario.door_closed} trash_has_garbage={self.scenario.trash_has_garbage}",
        ]
        self.draw_text_block(canvas, text, 10, 10, bg_color=(20, 20, 20), fg_color=(255, 255, 255))

    def draw_world_objects(self, canvas):
        objects = []
        if self.scenario.person_present:
            objects.append({"name": "person", "x": 2.2, "y": 0.2, "color": (40, 60, 220)})
        objects.append({"name": "door_closed" if self.scenario.door_closed else "door_open", "x": 2.8, "y": -0.6, "color": (20, 160, 20)})
        objects.append({"name": "trash_full" if self.scenario.trash_has_garbage else "trash_empty", "x": 1.8, "y": 0.9, "color": (80, 80, 80)})

        for obj in objects:
            rel_x, rel_z = self.world_to_camera(obj["x"], obj["y"])
            if rel_z <= 0.2:
                continue
            px = int(canvas.shape[1] * 0.5 + rel_x / max(rel_z, 1e-3) * 220)
            py = int(canvas.shape[0] * 0.68 - 80 / max(rel_z, 0.5))
            scale = max(24, int(120 / max(rel_z, 0.5)))
            self.draw_labeled_box(canvas, px, py, scale, obj["name"], obj["color"])

    def world_to_camera(self, obj_x, obj_y):
        dx = obj_x - self.x
        dy = obj_y - self.y
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        forward = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw
        return lateral, forward

    def draw_labeled_box(self, canvas, cx, cy, size, label, color):
        h, w = canvas.shape[:2]
        half_w = size // 2
        half_h = int(size * 0.75)
        x1 = max(0, cx - half_w)
        x2 = min(w - 1, cx + half_w)
        y1 = max(0, cy - half_h)
        y2 = min(h - 1, cy + half_h)
        canvas[y1:y2, x1:x2] = np.array(color, dtype=np.uint8)
        self.draw_text_block(canvas, [label], x1, max(0, y1 - 18), bg_color=(0, 0, 0), fg_color=(255, 255, 255), compact=True)

    def draw_text_block(self, canvas, lines, x, y, bg_color, fg_color, compact=False):
        char_w = 8
        line_h = 14 if compact else 18
        padding = 4
        width = min(canvas.shape[1] - x - 1, max(len(line) for line in lines) * char_w + padding * 2)
        height = min(canvas.shape[0] - y - 1, len(lines) * line_h + padding * 2)
        canvas[y:y + height, x:x + width] = np.array(bg_color, dtype=np.uint8)
        for i, line in enumerate(lines):
            self.draw_ascii_text(canvas, line[: max(1, (width - 2 * padding) // char_w)], x + padding, y + padding + i * line_h, fg_color)

    def draw_ascii_text(self, canvas, text, x, y, color):
        glyph_h = 10
        glyph_w = 6
        for idx, ch in enumerate(text):
            val = ord(ch)
            gx = x + idx * (glyph_w + 1)
            gy = y
            for row in range(glyph_h):
                for col in range(glyph_w):
                    if (val >> ((row + col) % 6)) & 1:
                        yy = gy + row
                        xx = gx + col
                        if 0 <= yy < canvas.shape[0] and 0 <= xx < canvas.shape[1]:
                            canvas[yy, xx] = np.array(color, dtype=np.uint8)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main():
    parser = argparse.ArgumentParser(description="ROS2 simulator for a Go2-like platform used by go2_vln_client.py")
    parser.add_argument("--image_topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--odom_topic", default="/sportmodestate")
    parser.add_argument("--cmd_topic", default="/api/sport/request")
    parser.add_argument("--publish_hz", type=float, default=15.0)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--person_present", type=str2bool, default=True)
    parser.add_argument("--door_closed", type=str2bool, default=True)
    parser.add_argument("--trash_has_garbage", type=str2bool, default=False)
    args = parser.parse_args()

    rclpy.init()
    node = Go2PlatformSim(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
