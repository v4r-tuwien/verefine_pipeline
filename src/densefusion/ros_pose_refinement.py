# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

"""
PoseRefinement.srv -- using DenseFusion
---------------------------------------
in:
    Detection det
    sensor_msgs/Image rgb
    sensor_msgs/Image depth
    PoseWithConfidence[] poses
out:
    PoseWithConfidence[] poses

"""

import numpy as np

import rospy
from object_detector_msgs.srv import refine_poses, refine_posesResponse

from src.util.dataset import YcbvDataset
from src.densefusion.densefusion import DenseFusion


if __name__ == "__main__":

    dataset = YcbvDataset()
    intrinsics = np.array([538.391033533567, 0.0, 315.3074696331638,
                           0.0, 538.085452058436, 233.0483557773859,
                           0.0, 0.0, 1.0]).reshape(3, 3)
    print("loading densefusion...")
    densefusion = DenseFusion(640, 480, intrinsics, dataset, mode="base")  # TODO set mode via parameter
    print("init ros...")
    rospy.init_node("poserefinement_densefusion")
    s = rospy.Service("refine_pose", refine_poses, densefusion.ros_refine)
    print("PoseRefinement with DenseFusion ready.")

    rospy.spin()
