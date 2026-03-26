
import numpy as np
import math
import time
import threading
import os
import sys
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import Quaternion, Pose, Point, Twist, Vector3

# 让d1_vln_client.py中的manager变量可用
from d1_vln_client import manager, control_thread, planning_thread, D1VlnManager

def create_fake_image_msg():
    # 创建一张虚拟图像消息
    msg = Image()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.height = 224
    msg.width = 224
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = 224 * 3
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    msg.data = arr.tobytes()
    return msg

def create_fake_odom_msg(x=1.0, y=2.0, yaw=0.5, vx=0.1, wz=0.05):
    msg = Odometry()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.pose.pose.position = Point(x=x, y=y, z=0.0)
    q = Quaternion()
    q.w = math.cos(yaw/2)
    q.z = math.sin(yaw/2)
    msg.pose.pose.orientation = q
    msg.twist.twist.linear = Vector3(x=vx, y=0.0, z=0.0)
    msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=wz)
    return msg

def main():
    rospy.init_node('fake_data_publisher')
    rgb_pub = rospy.Publisher('/odin1/image/undistorted', Image, queue_size=1)
    odom_pub = rospy.Publisher('/odin1/odometry_highfreq', Odometry, queue_size=1)
    global manager
    manager = D1VlnManager()
    # 先发布一帧，确保回调初始化
    rgb_msg = create_fake_image_msg()
    odom_msg = create_fake_odom_msg()
    rgb_pub.publish(rgb_msg)
    odom_pub.publish(odom_msg)
    
    t1 = threading.Thread(target=control_thread, daemon=True)
    t2 = threading.Thread(target=planning_thread, daemon=True)
    t1.start()
    t2.start()
    try:
        for _ in range(50):
            rgb_msg = create_fake_image_msg()
            odom_msg = create_fake_odom_msg()
            rgb_pub.publish(rgb_msg)
            odom_pub.publish(odom_msg)
            time.sleep(0.1)
        print('Fake data publishing finished.')
    except KeyboardInterrupt:
        print('Interrupted.')
    finally:
        print('Test finished.')

if __name__ == '__main__':
    main()
