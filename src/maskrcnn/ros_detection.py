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
from std_msgs.msg import Header
from object_detector_msgs.msg import BoundingBox, Detection, Detections
from object_detector_msgs.srv import detectron2_service_server
import ros_numpy

from util.dataset import YcbvDataset
from maskrcnn import MaskRcnnDetector


if __name__ == "__main__":

    dataset = YcbvDataset()
    maskrcnn = MaskRcnnDetector()

    def detect(req):
        print("request detection...")

        # === IN ===
        # --- rgb
        rgb = req.image
        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480
        rgb = ros_numpy.numpify(rgb)

        # === DETECTION ===
        obj_ids, rois, masks, scores = maskrcnn.detect(rgb)

        # === OUT ===
        detections = []
        for i, (obj_id, roi, score) in enumerate(zip(obj_ids, rois, scores)):

            detection = Detection()

            # ---
            name = ""
            for idx, obj_name in dataset.obj_names.items():
                if int(idx) == int(obj_id):
                    name = obj_name
                    break
            assert name != ""
            detection.name = name

            # ---
            bbox = BoundingBox()
            bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax = [int(val) for val in roi]  # TODO check
            detection.bbox = bbox

            # ---
            mask = masks[:, :, i]
            mask_ids = np.argwhere(mask.reshape((height * width)) > 0)
            detection.mask = list(mask_ids.flat)

            # ---
            detection.score = score

            detections.append(detection)

        ros_detections = Detections()
        ros_detections.width, ros_detections.height = 640, 480
        ros_detections.detections = detections

        return ros_detections


    rospy.init_node("detection_maskrcnn")
    s = rospy.Service("detect_objects", detectron2_service_server, detect)
    print("Detection with Mask R-CNN ready.")

    rospy.spin()
