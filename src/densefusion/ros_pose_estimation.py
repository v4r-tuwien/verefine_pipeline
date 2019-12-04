# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

"""
PoseEstimation.srv -- using DenseFusion
---------------------------------------
in:
    Detection det
    sensor_msgs/Image rgb
    sensor_msgs/Image depth
out:
    PoseWithConfidence[] poses

"""

import numpy as np

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from object_detector_msgs.msg import BoundingBox, Detection, PoseWithConfidence
from object_detector_msgs.srv import estimate_poses, estimate_posesResponse
import ros_numpy

from src.util.dataset import YcbvDataset
from src.densefusion.densefusion import DenseFusion


if __name__ == "__main__":

    dataset = YcbvDataset()
    densefusion = DenseFusion(640, 480, dataset, only_estimator=True)
    intrinsics = np.array([538.391033533567, 0.0, 315.3074696331638,
                           0.0, 538.085452058436, 233.0483557773859,
                           0.0, 0.0, 1.0]).reshape(3, 3)

    def estimate(req):
        print("pose estimate requested...")

        # === IN ===
        # --- rgb
        rgb = req.rgb
        width, height = rgb.width, rgb.height
        rgb = ros_numpy.numpify(rgb)

        # --- depth
        depth = req.depth
        depth = ros_numpy.numpify(depth)

        print(np.max(depth))

        # --- detection
        name = req.det.name
        obj_id = -1
        for idx, obj_name in dataset.obj_names.items():
            if obj_name == name:
                obj_id = idx + 1
                break
        assert obj_id > 0  # should start from 1

        bbox = req.det.bbox
        obj_roi = [bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax]

        mask_ids = req.det.mask
        print(len(mask_ids))
        mask_ids = np.array(mask_ids)
        print(mask_ids.shape)
        obj_mask = np.zeros((height * width), dtype=np.uint8)
        print(obj_mask.shape)
        obj_mask[mask_ids] = 1
        obj_mask = obj_mask.reshape((height, width))

        # === POSE ESTIMATION ===
        hypotheses_per_instance = 5
        hypotheses = densefusion.estimate(rgb, depth, intrinsics, obj_roi, obj_mask, obj_id,
                                          hypotheses_per_instance=hypotheses_per_instance)
        assert len(hypotheses) == hypotheses_per_instance

        # === OUT ===
        poses = []
        for r, t, c in hypotheses:
            pose = PoseWithConfidence()

            # --- name
            pose.name = name

            # --- pose
            pose.pose = Pose()
            pose.pose.position = ros_numpy.msgify(Point, t)
            r = np.concatenate((r[1:],[r[0]]))  # ros_numpy expects x, y, z, w
            pose.pose.orientation = ros_numpy.msgify(Quaternion, r)

            # --- confidence
            pose.confidence = c

            poses.append(pose)

        response = estimate_posesResponse()
        response.poses = poses
        return response

    rospy.init_node("poseestimation_densefusion")
    s = rospy.Service("estimate_pose", estimate_poses, estimate)
    print("PoseEstimation with DenseFusion ready.")

    rospy.spin()
