import numpy as np
import rospy
import ros_numpy
import actionlib

from PIL import Image as PImage
import json
import gc

import time
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

from sensor_msgs.msg import Image, RegionOfInterest
from object_detector_msgs.srv import estimate_poses
from object_detector_msgs.msg import Detections, Detection, BoundingBox
from robokudo_msgs.msg import GenericImgProcAnnotatorAction, GenericImgProcAnnotatorResult, GenericImgProcAnnotatorFeedback

# === define pipeline clients ===

# def detect(rgb):
#     rospy.wait_for_service('detect_objects')
#     try:
#         detect_objects = rospy.ServiceProxy('detect_objects', detectron2_service_server)
#         response = detect_objects(rgb)
#         return response.detections.detections
#     except rospy.ServiceException as e:
#         print("Service call failed: %s" % e)


def estimate(rgb, depth, detection):
    rospy.wait_for_service('estimate_poses')
    try:
        estimate_pose = rospy.ServiceProxy('estimate_poses', estimate_poses)
        response = estimate_pose(detection, rgb, depth)
        return response.poses
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


RGB_TOPIC = rospy.get_param('/pose_estimator/color_topic')
DEPTH_TOPIC = rospy.get_param('/pose_estimator/depth_topic')
CAMERA_INFO = rospy.get_param('/pose_estimator/camera_info_topic')

class GraspPoseEstimator:

    def __init__(self):
        self.server = actionlib.SimpleActionServer('find_grasppose', GenericImgProcAnnotatorAction, self.find_grasppose,
                                                   auto_start=False)
        self.server.start()
        self.observation_mask = False
        self.current_poses = []
        self.segmask_publisher = rospy.Publisher('/pose_estimator/segmentation', Image)

    def find_grasppose(self, goal):
        result = GenericImgProcAnnotatorResult()
        result.success = False
        result.result_feedback = "calculated: "
        feedback = GenericImgProcAnnotatorFeedback()

        # height and width of the image needed
        width, height = goal.rgb.width, goal.rgb.height

        # === check if we have an image ===
        if goal.rgb is None or goal.depth is None:
            print("no images available")
            result.result_feedback = "no images available"
            result.success = False
            self.server.set_succeeded(result)
            # self.server.set_preempted()
            return

        # === run pipeline ===
        gc.collect()

        self.current_poses = []

        with open("/task/src/config.json", 'r') as file:
            config = json.load(file)
        th_detection = config["filter"]["detection_min_score"]
        th_estimation = config["filter"]["estimation_min_score"]
        th_refinement = config["filter"]["refinement_min_score"]
        print("settings:\n   thresholds: detection=%0.3f, estimation=%0.3f, refinement=%0.3f"
              % (th_detection, th_estimation, th_refinement))

        # detect all instances in image
        #feedback.feedback = "requesting detection..."
        #self.server.publish_feedback(feedback)
        #print("requesting detection...")
        #st = time.time()
        #detections = detect(goal.rgb)
        #duration_detection = time.time() - st
        #feedback.feedback = "received detection."
        #self.server.publish_feedback(feedback)
        #print("   received detection.")

        if (goal.mask_detections is None or len(goal.mask_detections) == 0) and (goal.bb_detections is None or len(goal.bb_detections) == 0):
            print("nothing detected")
            result.result_feedback = "nothing detected"
            result.success = False
            self.server.set_succeeded(result)
            return
        
        detections = []
        for index, class_name in enumerate(goal.class_names):
            detection = Detection()
            detection.name = class_name

            bbox = BoundingBox()
            if len(goal.mask_detections) > index:
                img = ros_numpy.numpify(goal.mask_detections[index])
                detection.mask = np.argwhere(img.flatten() > 0)
                rows = np.any(img, axis=1)
                cols = np.any(img, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                bbox.xmin = cmin
                bbox.xmax = cmax
                bbox.ymin = rmin
                bbox.ymax = rmax
            elif len(goal.bb_detections) > index:
                bbox.xmin = goal.bb_detections[index].x_offset
                bbox.xmax = goal.bb_detections[index].x_offset + goal.bb_detections[index].width
                bbox.ymin = goal.bb_detections[index].y_offset
                bbox.ymax = goal.bb_detections[index].y_offset + goal.bb_detections[index].height
                for col_index in range(bbox.xmin, bbox.xmax):
                    for row_index in range(bbox.ymin, bbox.ymax):
                        detection.mask.append(row_index * width + col_index)
                detection.mask = sorted(detection.mask)
            else:
                print("mask or boundingbox error")
                result.result_feedback = "mask or boundingbox error"
                result.success = False
                self.server.set_succeeded(result)
                return

            detection.bbox = bbox            
            detections.append(detection)


        # fill result with the detections
        mask_detected_objects = np.ones((height * width), dtype=np.uint8)
        # -- Bounding Box -- & -- object segmentation mask -- & -- class ID --
        obj_mask = np.zeros((height * width), dtype=np.uint8)
        seg_image = ros_numpy.numpify(goal.rgb).copy()
        length = len(detections)
        for idx, detection in enumerate(detections):
            result.bounding_boxes.append(RegionOfInterest(detection.bbox.xmin, detection.bbox.ymin, detection.bbox.ymax
                                                          - detection.bbox.ymin, detection.bbox.xmax
                                                          - detection.bbox.xmin, False))
            result.descriptions.append(detection.name)
            # object segmentation mask
            mask_ids = detection.mask
            mask_ids = np.array(mask_ids)
            obj_mask[mask_ids] = 1
            colors = plt.get_cmap('magma')((idx+0.01)/length if length>0 else 1)
            colors = np.array((colors[0]*256, colors[1]*256, colors[2]*256), dtype=np.uint8)
            seg_mask = np.zeros((height*width), dtype=bool)
            seg_mask[mask_ids] = True 
            seg_mask = seg_mask.reshape((height, width))
            seg_image[seg_mask] = colors[:3] 

            # class ID
            name = detection.name
            try:
                f = open("/verefine/data/ycbv_names.json")
                ycbv_names = json.load(f)
                f.close()
            except:
                print("YCBV_names not found")

            obj_id = -1
            for number in ycbv_names:
                if ycbv_names[number] == name:
                    obj_id = int(number) 
                    break
            assert obj_id > 0  # should start from 1

            # class ID of the detection
            result.class_ids.append(obj_id)

        result.result_feedback = result.result_feedback + "bounding_boxes, class_ids, class_confidences, descriptions"
        self.segmask_publisher.publish(ros_numpy.image.numpy_to_image(seg_image, encoding='rgb8'))
        # -- object segmentation Image --
        result.image = goal.rgb
        seg_image = ros_numpy.numpify(goal.rgb).copy()
        obj_mask = obj_mask.reshape((height, width))
        seg_image[obj_mask == 0, 0] = 0
        seg_image[obj_mask == 0, 1] = 0
        seg_image[obj_mask == 0, 2] = 0
        result.image.data = seg_image.flatten().tolist()
        result.result_feedback = result.result_feedback + ", image"

        # -- for each detection/instance calculate the pose --
        poses = []
        confidences = []
        durations = []
        current_score = -1
        feedback.feedback = "requesting pose estimate and refine step..."
        self.server.publish_feedback(feedback)
        for detection in detections:

            # estimate a set of candidate posesposs
            print("requesting pose estimate and refine step...")
            st = time.time()
            instance_poses = estimate(goal.rgb, goal.depth, detection)

            duration = time.time() - st
            print("   received refined poses.")
            if instance_poses is None or len(instance_poses) == 0:
                print("all poses for %s rejected after refinement" % detection.name)
                continue

            # # reject by pose confidence
            print(",".join(["%0.3f" % pose.confidence for pose in instance_poses]))
            instance_poses = [pose for pose in instance_poses if pose.confidence > th_refinement]
            # add to set of poses of detected instances
            if len(instance_poses) > 0:
                poses += instance_poses
                confidences += [pose.confidence for pose in instance_poses]
                durations += [duration] * len(instance_poses)
            else:
                print("all poses of %s rejected" % detection.name)
        assert len(poses) == len(confidences)

        # === select pose with highest confidence === (TODO and scale by distance?)
        if len(confidences) > 0:
            best_hypothesis = np.argmax(confidences)
            best_pose = poses[best_hypothesis]
            # self.vis_pose(poses) #bestpose) # not working, some thirdpart-tool error

            self.current_poses = poses
        else:
            print("no valid poses")
            result.result_feedback = "no valid poses"
            result.success = False
            self.server.set_succeeded(result)
            # self.server.set_preempted()
            return

        feedback.feedback = "poses calculated"
        self.server.publish_feedback(feedback)

        # -- Pose result --
        for index, pose_with_confidence in enumerate(poses):
            result.pose_results.append(pose_with_confidence.pose)
            result.class_confidences.append(confidences[index])
        result.result_feedback = result.result_feedback + ", pose_results"

        # Notify client that action is finished and result is ready
        result.success = True
        self.server.set_succeeded(result)
        return


if __name__ == "__main__":
    rospy.init_node("pose_estimation_actionserver")

    grasper = GraspPoseEstimator()

    print("pose_estimation_actionserver test ready!")

    rospy.spin()
    