import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class RemoteJoyListener:
    def __init__(self):
        self.pub = rospy.Publisher('/remote_cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/joy', Joy, self.joy_cb)
    def joy_cb(self, msg):
        # 假设axes[1]为前后, axes[0]为左右
        t = Twist()
        t.linear.x = msg.axes[1] * 0.3
        t.angular.z = msg.axes[0] * 0.5
        self.pub.publish(t)
        print(f'Remote joy: v={t.linear.x:.2f}, w={t.angular.z:.2f}')

if __name__ == "__main__":
    rospy.init_node('remote_joy_listener')
    RemoteJoyListener()
    rospy.spin()
