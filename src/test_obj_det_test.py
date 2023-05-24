#! /usr/bin/env python3
import rospy
from robokudo_msgs.msg import GenericImgProcAnnotatorAction, GenericImgProcAnnotatorGoal
import actionlib
from sensor_msgs.msg import Image, RegionOfInterest
from object_detector_msgs.srv import detectron2_service_server
import copy
import numpy as np

def detect(rgb):
    rospy.wait_for_service('detect_objects')
    try:
        detect_objects = rospy.ServiceProxy('detect_objects', detectron2_service_server)
        response = detect_objects(rgb)
        return response.detections.detections
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def get_poses():
    client = actionlib.SimpleActionClient('/pose_estimator/find_grasppose', GenericImgProcAnnotatorAction)
    res = client.wait_for_server(rospy.Duration(10.0))
    if res is False:
        rospy.logerr('Timeout when trying to connect to actionserver')
        return
    goal = GenericImgProcAnnotatorGoal()
    print('Waiting for images')
    rgb = rospy.wait_for_message('/hsrb/head_rgbd_sensor/rgb/image_rect_color', Image)
    depth = rospy.wait_for_message('/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw', Image)

    print('Waiting for detection')
    detections = detect(rgb)
    print("   received detection.")
    if detections is None or len(detections) == 0:
        print("nothing detected")
        exit(1)
    
    bb_detections = []
    mask_detections = []
    class_names = []
    for det in detections:
        bb_detection = RegionOfInterest()
        bb_detection.width = det.bbox.xmax - det.bbox.xmin
        bb_detection.height = det.bbox.ymax - det.bbox.ymin
        bb_detection.x_offset = det.bbox.xmin
        bb_detection.y_offset = det.bbox.ymin
        bb_detections.append(bb_detection)

        mask_detection = copy.deepcopy(rgb)
        data = np.flatten(np.zeros((rgb.width, rgb.height), dtype=np.uint8))
        data[det.mask] = 255
        mask_detection.data = data.reshape((rgb.width, rgb.height))
        mask_detections.append(mask_detection)

        class_names.append(det.name)


    print('Sending Goal')
    goal.rgb = rgb
    goal.depth = depth
    goal.bb_detections = bb_detections
    goal.mask_detections = mask_detections
    goal.class_names = class_names
    client.send_goal(goal)
    client.wait_for_result()
    print(f"Got pose for: {client.get_result().descriptions}")

if __name__ == "__main__":
    rospy.init_node("get_poses")
    get_poses()

