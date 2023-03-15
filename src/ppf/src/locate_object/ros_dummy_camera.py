import rospy
import ros_numpy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import numpy as np
import imageio
import os
import yaml


class DummyCamera:
    def __init__(self):
        self.rate = rospy.Rate(1)
        self.pub_color = rospy.Publisher(rospy.get_param('/locateobject/color_topic'), Image, queue_size=10)
        self.pub_depth = rospy.Publisher(rospy.get_param('/locateobject/depth_topic'), Image, queue_size=10)
        self.pub_info = rospy.Publisher(rospy.get_param('/locateobject/camera_info_topic'), CameraInfo, queue_size=10)

        # get images
        self.observation_dir = "/canister/data/canister-painted"
        self.observations = []
        for fi in range(1, 4):
            # assuming we get an RGBD image as input - both are float arrays, rgb is in range [0,1] and depth in [mm]
            rgb = imageio.imread(os.path.join(self.observation_dir, f"color_{fi:03d}.png"))[..., :3].astype(np.uint8)
            rgb = ros_numpy.msgify(Image, rgb, encoding="rgb8")
            depth = imageio.imread(os.path.join(self.observation_dir, f"depth_{fi:03d}.png")).astype(np.float32)
            depth = ros_numpy.msgify(Image, depth, encoding="32FC1")
            mask = None
            self.observations.append([rgb, depth, mask])

        # get camera info
        calib_data = yaml.load(open("/canister/data/dummy_camera.yml", 'r'), Loader=yaml.FullLoader)
        camera_info_msg = CameraInfo()
        camera_info_msg.width = calib_data["image_width"]
        camera_info_msg.height = calib_data["image_height"]
        camera_info_msg.K = calib_data["camera_matrix"]["data"]
        camera_info_msg.D = calib_data["distortion_coefficients"]["data"]
        camera_info_msg.R = calib_data["rectification_matrix"]["data"]
        camera_info_msg.P = calib_data["projection_matrix"]["data"]
        camera_info_msg.distortion_model = calib_data["distortion_model"]
        self.camera_info = camera_info_msg

    def start(self):
        i, seq = 0, 0
        # rospy.spin()
        while not rospy.is_shutdown():
            header = Header()
            header.seq = seq
            header.stamp = rospy.Time.now()
            header.frame_id = "d435"
            rgb, depth, _ = self.observations[i]
            rgb.header = header
            depth.header = header
            self.pub_color.publish(rgb)
            self.pub_depth.publish(depth)
            self.pub_info.publish(self.camera_info)

            self.rate.sleep()
            i, seq = (i + 1) % len(self.observations), seq + 1


if __name__ == '__main__':
    rospy.init_node("camera", anonymous=True)
    publisher = DummyCamera()
    publisher.start()
