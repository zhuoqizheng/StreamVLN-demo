
import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import requests
import os
import time
import io
import json
import numpy as np
from PIL import Image as PIL_Image

policy_init = True

def eval_vln(image, instruction=None, url=None):
    global policy_init
    if url is None:
        url = os.environ.get('STREAMVLN_SERVER_URL', 'http://localhost:5801/eval_vln')
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
    print(f"total time(delay + policy): {time.time() - start}")
    print(response.text)
    action = json.loads(response.text)['action']
    return action

def get_ros_image(rgb_topic, timeout=5):
    bridge = CvBridge()
    msg = rospy.wait_for_message(rgb_topic, RosImage, timeout=timeout)
    img = bridge.imgmsg_to_cv2(msg, 'bgr8')
    img = img[:, :, ::-1]  # BGR to RGB
    return img

if __name__ == "__main__":
    rospy.init_node('vln_client_upload_image')
    url = os.environ.get('STREAMVLN_SERVER_URL', 'http://localhost:5801/eval_vln')
    rgb_topic = os.environ.get("ODIN_RGB_TOPIC", os.environ.get("D1_RGB_TOPIC", "/odin1/image/undistorted"))
    instruction = os.environ.get('STREAMVLN_INSTRUCTION', None)
    print(f"Using topic: {rgb_topic}, url: {url}")
    while not rospy.is_shutdown():
        try:
            img = get_ros_image(rgb_topic)
            action = eval_vln(img, instruction, url)
            print('Received action:', action)
        except Exception as e:
            print('Error:', e)
        time.sleep(1)
