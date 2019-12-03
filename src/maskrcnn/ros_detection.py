# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

"""
detectron2_service_server.srv -- using Matterport's Mask R-CNN implementation
---------------------------------------
in:
    sensor_msgs/Image image
out:
    object_detector_msgs/Detections detections

"""

import numpy as np

import rospy
from object_detector_msgs.msg import BoundingBox, Detection
from object_detector_msgs.srv import detectron2_service_server
import ros_numpy

from util.dataset import YcbvDataset
from maskrcnn import MaskRcnnDetector


if __name__ == "__main__":

    dataset = YcbvDataset()
    maskrcnn = MaskRcnnDetector()

    def detect(req):
        # === IN ===
        # --- rgb
        rgb = req.rgb
        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480
        rgb = ros_numpy.numpify(rgb)

        # === DETECTION ===
        obj_ids, rois, masks, scores = maskrcnn.detect(rgb)

        # === OUT ===
        detections = []
        for obj_id, roi, mask, score in zip(obj_ids, rois, masks, scores):

            detection = Detection()

            # ---
            name = ""
            for idx, obj_name in dataset.obj_names.items():
                if idx == obj_id - 1:
                    name = obj_name
                    break
            assert name != ""
            detection.name = name

            # ---
            bbox = BoundingBox()
            bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax = roi  # TODO check
            detection.bbox = bbox

            # ---
            mask_ids = np.argwhere(mask.reshape((height * width)) > 0)
            detection.mask = mask_ids

            # ---
            detection.score = score

            detection.append(detection)

        return detections


    rospy.init_node("detection_maskrcnn")
    s = rospy.Service("detectron2_service_server", detectron2_service_server, detect)
    print("Detection with Mask R-CNN ready.")

    rospy.spin()
