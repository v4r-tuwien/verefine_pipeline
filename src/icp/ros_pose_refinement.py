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
from geometry_msgs.msg import Pose, Point, Quaternion
from object_detector_msgs.msg import BoundingBox, Detection, PoseWithConfidence
from object_detector_msgs.srv import refine_poses
import ros_numpy

from util.dataset import YcbvDataset
from icp.icp import Icp

if __name__ == "__main__":

    dataset = YcbvDataset()
    intrinsics = np.array([538.391033533567, 0.0, 315.3074696331638,
                           0.0, 538.085452058436, 233.0483557773859,
                           0.0, 0.0, 1.0]).reshape(3, 3)
    icp = Icp(dataset)

    def refine(req):
        # === IN ===
        # --- rgb
        rgb = req.rgb
        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480
        rgb = ros_numpy.numpify(rgb)

        # --- depth
        depth = req.depth
        depth = ros_numpy.numpify(depth)

        # --- detection
        name = req.detection.name
        obj_id = -1
        for idx, obj_name in dataset.obj_names.items():
            if obj_name == name:
                obj_id = idx + 1
                break
        assert obj_id > 0  # should start from 1

        bbox = req.detection.bbox
        obj_roi = [bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax]

        mask_ids = req.detection.mask
        obj_mask = np.zeros((height * width), dtype=np.uint8)
        obj_mask[mask_ids] = 1
        obj_mask = obj_mask.reshape((height, width))

        # --- estimates
        estimates = req.poses

        # === POSE REFINEMENT ===
        iterations = 2
        refined = []
        for estimate in estimates:
            estimate = [
                ros_numpy.numpify(estimate.quaternion),
                ros_numpy.numpify(estimate.point),
                estimate.confidence
            ]
            hypothesis = icp.refine(rgb, depth, intrinsics, obj_roi, obj_mask, obj_id,
                                    estimate, iterations)
            refined.append(hypothesis)
        assert len(refined) == len(estimates)

        # === OUT ===
        poses = []
        for r, t, c in refined:
            pose = PoseWithConfidence()

            # --- name
            pose.name = name

            # --- pose
            pose.pose = Pose()
            pose.pose.point = ros_numpy.msgify(Point, t, hom=False)
            r = np.concatenate((r[1:], [r[0]]))  # ros_numpy expects x, y, z, w
            pose.pose.quaternion = ros_numpy.msgify(Quaternion, r)

            # --- confidence
            pose.confidence = c

            poses.append(pose)

        return poses


    rospy.init_node("poserefinement_icp")
    s = rospy.Service("refine_pose", refine_poses, refine)
    print("PoseRefinement with ICP ready.")

    rospy.spin()
