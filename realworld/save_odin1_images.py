#!/usr/bin/env python3
import argparse
import os
import time
from typing import Dict

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image


TOPIC_SPECS = {
    "/odin1/image": "raw",
    "/odin1/image/compressed": "compressed",
    "/odin1/image/intensity_gray": "raw",
    "/odin1/image/undistorted": "raw",
    "/odin1/image/undistorted/compressed": "compressed",
}


def _topic_to_dirname(topic: str) -> str:
    return topic.strip("/").replace("/", "_")


class Odin1ImageSaver:
    def __init__(self, output_dir: str, max_frames_per_topic: int):
        rospy.init_node("odin1_image_saver", anonymous=False)
        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.max_frames_per_topic = max_frames_per_topic
        self.counts: Dict[str, int] = {topic: 0 for topic in TOPIC_SPECS}
        self.subs = []

        os.makedirs(self.output_dir, exist_ok=True)
        rospy.loginfo("Saving images to: %s", self.output_dir)

        for topic, mode in TOPIC_SPECS.items():
            topic_dir = os.path.join(self.output_dir, _topic_to_dirname(topic))
            os.makedirs(topic_dir, exist_ok=True)

            if mode == "compressed":
                sub = rospy.Subscriber(
                    CompressedImage,
                    topic,
                    lambda msg, t=topic: self._on_compressed(msg, t),
                    queue_size=10,
                )
            else:
                sub = rospy.Subscriber(
                    Image,
                    topic,
                    lambda msg, t=topic: self._on_raw(msg, t),
                    queue_size=10,
                )

            self.subs.append(sub)
            rospy.loginfo("Subscribed: %s (%s)", topic, mode)

    def _build_path(self, topic: str, ext: str) -> str:
        topic_dir = os.path.join(self.output_dir, _topic_to_dirname(topic))
        now_ms = int(time.time() * 1000)
        idx = self.counts[topic]
        name = f"{idx:06d}_{now_ms}.{ext}"
        return os.path.join(topic_dir, name)

    def _try_shutdown(self):
        if self.max_frames_per_topic <= 0:
            return

        done = all(c >= self.max_frames_per_topic for c in self.counts.values())
        if done:
            rospy.loginfo("Reached max frames for all topics. Shutting down.")
            rospy.signal_shutdown("capture complete")

    def _on_raw(self, msg: Image, topic: str):
        if self.max_frames_per_topic > 0 and self.counts[topic] >= self.max_frames_per_topic:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if cv_img.ndim == 2:
                out = cv_img
            else:
                # Keep OpenCV convention (BGR) for color images.
                out = cv_img

            path = self._build_path(topic, "png")
            ok = cv2.imwrite(path, out)
            if not ok:
                rospy.logerr("Failed to save image: %s", path)
                return

            self.counts[topic] += 1
            rospy.loginfo("Saved [%s] %s", topic, path)
            self._try_shutdown()
        except Exception as exc:
            rospy.logerr("Raw callback failed for %s: %s", topic, exc)

    def _on_compressed(self, msg: CompressedImage, topic: str):
        if self.max_frames_per_topic > 0 and self.counts[topic] >= self.max_frames_per_topic:
            return

        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if cv_img is None:
                rospy.logerr("Failed to decode compressed image from %s", topic)
                return

            fmt = (msg.format or "").lower()
            ext = "jpg" if "jpeg" in fmt or "jpg" in fmt else "png"
            path = self._build_path(topic, ext)
            ok = cv2.imwrite(path, cv_img)
            if not ok:
                rospy.logerr("Failed to save image: %s", path)
                return

            self.counts[topic] += 1
            rospy.loginfo("Saved [%s] %s", topic, path)
            self._try_shutdown()
        except Exception as exc:
            rospy.logerr("Compressed callback failed for %s: %s", topic, exc)


def parse_args():
    parser = argparse.ArgumentParser(description="Subscribe odin1 ROS1 image topics and save frames.")
    parser.add_argument(
        "--output-dir",
        default="./odin1_captures",
        help="Directory to save images. A subfolder is created per topic.",
    )
    parser.add_argument(
        "--max-frames-per-topic",
        type=int,
        default=20,
        help="Stop after N frames per topic. Use <=0 to run continuously.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Odin1ImageSaver(
        output_dir=os.path.abspath(args.output_dir),
        max_frames_per_topic=args.max_frames_per_topic,
    )
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
