import rospy
from geometry_msgs.msg import Twist

class RemoteMuxNode:
    def __init__(self):
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/remote_cmd_vel', Twist, self.remote_cb)
        rospy.Subscriber('/auto_cmd_vel', Twist, self.auto_cb)
        self.last_remote = rospy.Time(0)
        self.remote_msg = None
        self.auto_msg = None
    def remote_cb(self, msg):
        self.last_remote = rospy.Time.now()
        self.remote_msg = msg
        self.publish()
    def auto_cb(self, msg):
        self.auto_msg = msg
        self.publish()
    def publish(self):
        now = rospy.Time.now()
        if self.remote_msg and (now - self.last_remote).to_sec() < 2.0:
            self.pub.publish(self.remote_msg)
            print('Mux: remote_cmd_vel')
        elif self.auto_msg:
            self.pub.publish(self.auto_msg)
            print('Mux: auto_cmd_vel')

if __name__ == "__main__":
    rospy.init_node('remote_mux_node')
    RemoteMuxNode()
    rospy.spin()
