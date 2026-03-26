import rospy
from geometry_msgs.msg import Twist
import time

if __name__ == "__main__":
    rospy.init_node('cmdvel_test')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    stop_duration = 1.0  # 秒，停止命令持续时间
    while not rospy.is_shutdown():
        try:
            x = float(input('输入线速度x (m/s): '))
            z = float(input('输入角速度z (rad/s): '))
        except Exception:
            print('输入无效，重试')
            continue
        # 发送一次用户指定速度
        msg = Twist()
        msg.linear.x = x
        msg.angular.z = z
        pub.publish(msg)
        print(f'Published cmd_vel: v={x}, w={z}')
        time.sleep(0.2)
        # 发送一段时间的0速度，确保停止
        stop_msg = Twist()
        t0 = time.time()
        while time.time() - t0 < stop_duration and not rospy.is_shutdown():
            pub.publish(stop_msg)
            time.sleep(0.05)
        print('Published stop cmd_vel (v=0, w=0)')
