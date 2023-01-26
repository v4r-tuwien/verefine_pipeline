# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import numpy as np
from PIL import Image as Img
import os

import rospy
import ros_numpy
from sensor_msgs.msg import Image
from verefine_msgs.srv import detect_objects, estimate_poses, visualize_ycbv, visualize_ycbvResponse

BASEPATH = "/ycbv/test/"


if __name__ == "__main__":

    def detect(rgb, depth):
        rospy.wait_for_service('detect_objects')
        try:
            srv_detect_objects = rospy.ServiceProxy('detect_objects', detect_objects)
            response = srv_detect_objects(rgb, depth)
            return response.detections
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


    def estimate(rgb, depth, detections, mode):
        rospy.wait_for_service('estimate_poses')
        try:
            srv_estimate_poses = rospy.ServiceProxy('estimate_poses', estimate_poses)
            response = srv_estimate_poses(rgb, depth, detections, mode)
            return response.poses
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


    def visualize(req):
        scene, frame = int(req.scene), int(req.frame)
        print(f"visualize scene {scene:04d} / frame {frame:06d}...")
        response = visualize_ycbvResponse()
        if not os.path.exists(os.path.join(BASEPATH, f"{scene:06d}/rgb/{frame:06d}.png")):
            print("  does not exist")
            return response

        # load images from YCB-Video
        rgb = np.asarray(Img.open(os.path.join(BASEPATH, f"{scene:06d}/rgb/{frame:06d}.png")))
        depth = np.asarray(
            Img.open(os.path.join(BASEPATH, f"{scene:06d}/depth/{frame:06d}.png"))) / 10000.0  # to meters

        # convert images to ROS format
        img_rgb = ros_numpy.msgify(Image, rgb, encoding="rgb8")
        img_depth = ros_numpy.msgify(Image, depth.astype(np.float32), encoding="32FC1")

        # run detection and pose estimation
        detections = detect(img_rgb, img_depth)
        poses = estimate(img_rgb, img_depth, detections, int(req.mode))

        return response

    rospy.init_node("verefine_visualization")
    s = rospy.Service("visualize_ycbv", visualize_ycbv, visualize)
    print("Visualization ready.")

    rospy.spin()
