import numpy as np
import rospy
import ros_numpy
import actionlib
import tf

import os
import PIL
from PIL import Image as PImage
import json
import gc

import time
from datetime import datetime

import cv2
import scipy.spatial.transform as scit

import message_filters
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Empty, EmptyResponse
from object_detector_msgs.msg import PoseWithConfidence
from object_detector_msgs.srv import detectron2_service_server, estimate_poses, refine_poses, get_poses, get_posesResponse
#from grasping_pipeline.msg import ExecuteGraspAction, ExecuteGraspGoal
#from sasha_handover.msg import HandoverAction, HandoverGoal

from util.renderer import EglRenderer as Renderer
from util.dataset import YcbvDataset


# === define pipeline clients ===

def detect(rgb):
    rospy.wait_for_service('detect_objects')
    try:
        detect_objects = rospy.ServiceProxy('detect_objects', detectron2_service_server)
        response = detect_objects(rgb)
        return response.detections.detections
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def estimate(rgb, depth, detection):
    rospy.wait_for_service('estimate_poses')
    try:
        estimate_pose = rospy.ServiceProxy('estimate_poses', estimate_poses)
        response = estimate_pose(detection, rgb, depth)
        return response.poses
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def refine(rgb, depth, detection, poses):
    rospy.wait_for_service('refine_pose')
    try:
        refine_pose = rospy.ServiceProxy('refine_pose', refine_poses)
        response = refine_pose(detection, rgb, depth, poses)
        return response.poses
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


# === define grasp service ===
RGB_TOPIC = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
DEPTH_TOPIC = "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw"
CAMERA_INFO = "/hsrb/head_rgbd_sensor/rgb/camera_info"


class Grasper:

    def __init__(self):
        self.rgb = None
        self.depth = None
        self.working = False

        self.dataset = YcbvDataset()
        width, height, intrinsics = self.dataset.width, self.dataset.height, self.dataset.camera_intrinsics
        self.renderer = Renderer(self.dataset, width, height)

        self.pub_segmentation = rospy.Publisher("/hsr_grasping/segmentation", Image)
        # self.pub_initial = rospy.Publisher("/hsr_grasping/initial_poses", Image)
        self.pub_refined = rospy.Publisher("/hsr_grasping/refined_poses", Image)
        self.pub_poses = tf.TransformBroadcaster()
        self.current_poses = []

    def rgbd_callback(self, rgb, depth):
        # print("callback...")
        if not self.working:
            # print("   set images")
            self.rgb, self.depth = rgb, depth
            for pose in self.current_poses:
               self.tf_pose(pose, "")

    def grasp(self, req):

        response = get_posesResponse()

        # === check if we have an image ===

        if self.rgb is None or self.depth is None:
            print("no images available")
            return response

        # === run pipeline ===

        self.working = True
        gc.collect()

        #self.renderer._create_egl_context()  # TODO needed?

        self.current_poses = []

        with open("/task/src/config.json", 'r') as file:
            config = json.load(file)
        th_detection = config["filter"]["detection_min_score"]
        th_estimation = config["filter"]["estimation_min_score"]
        th_refinement = config["filter"]["refinement_min_score"]
        print("settings:\n   thresholds: detection=%0.3f, estimation=%0.3f, refinement=%0.3f"
            % (th_detection, th_estimation, th_refinement))

        # detect all instances in image
        print("requesting detection...")
        st = time.time()
        detections = detect(self.rgb)
        duration_detection = time.time() - st
        print("   received detection.")
        if detections is None or len(detections) == 0:
            print("nothing detected")
            self.working = False
            return response
        self.vis_detect(detections)

        # for each instance...
        poses = []
        confidences = []
        durations = []
        for detection in detections:
            # reject based on detection score
            if detection.score < th_detection:
                print("detection of %s rejected" % detection.name)
                continue

            # estimate a set of candidate poses
            print("requesting pose estimate...")
            st = time.time()
            instance_poses = estimate(self.rgb, self.depth, detection)
            print("   received pose.")
            #instance_poses = [pose for pose in instance_poses if pose.confidence > th_estimation]
            #if len(instance_poses) == 0:
            #    print("all poses for %s rejected after estimation" % detection.name)
            #    continue
            # assert len(instance_poses) == 5  # TODO
            # for instance_pose in instance_poses:
            #   self.vis_pose(instance_poses, "_estimated")

            # refine candidate poses
            #print("requesting pose refinement...")
            #instance_poses = refine(self.rgb, self.depth, detection, instance_poses)
            duration = time.time() - st
            print("   received refined poses.")
            if len(instance_poses) == 0:
                print("all poses for %s rejected after refinement" % detection.name)
                continue
            # for instance_pose in instance_poses:
            #   self.vis_pose(instance_poses, "_refined")

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

        # === to numpy and store
        if not os.path.exists("/task/lm"):
            os.mkdir("/task/lm")
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        PImage.fromarray(ros_numpy.numpify(self.rgb)).save("/task/lm/%s_rgb.png" % (filename))
        PImage.fromarray(ros_numpy.numpify(self.depth)).save("/task/lm/%s_depth.png" % (filename))

        for ii, detection in enumerate(detections):
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
                    obj_id = int(number) #+ 1 #?
                    break
            assert obj_id > 0  # should start from 1

            mask_ids = detection.mask
            mask_ids = np.array(mask_ids)
            width, height = self.rgb.width, self.rgb.height
            obj_mask = np.zeros((height * width), dtype=np.uint8)
            obj_mask[mask_ids] = 255
            obj_mask = obj_mask.reshape((height, width))

            PImage.fromarray(obj_mask).save("/task/lm/%s-%i_mask.png" % (filename, ii))

        for pose, duration in zip(poses, durations):
            pose_estimate = [
                ros_numpy.numpify(pose.pose.orientation),
                ros_numpy.numpify(pose.pose.position),
                pose.confidence
            ]

            from scipy.spatial.transform import Rotation
            T_obj = np.matrix(np.eye(4))
            T_obj[:3, :3] = Rotation.from_quat(pose_estimate[0]).as_dcm()
            T_obj[:3, 3] = pose_estimate[1].reshape(3, 1)

            with open("/task/lm/%s.csv" % filename, 'a') as file:
                parts = [str(pose.name),
                                    "%0.3f" % pose_estimate[2],
                                    " ".join(["%0.6f" % v for v in np.array(T_obj[:3, :3]).reshape(9)]),
                                    " ".join(["%0.6f" % (v * 1000) for v in np.array(T_obj[:3, 3])]),
                                    "%0.3f" % (duration + duration_detection)]
                file.write(",".join(parts) + "\n")

        # === select pose with highest confidence === TODO and scale by distance?

        if len(confidences) > 0:
            best_hypothesis = np.argmax(confidences)
            best_pose = poses[best_hypothesis]
            #self.vis_pose(poses) #bestpose)

            self.current_poses = poses
        else:
            print("no valid poses")
            self.working = False
            return response

        # === annotate grasp ===

        # TODO annotate and select

        # grasp_pose_stamped = PoseStamped()
        # grasp_pose_stamped.header = self.rgb.header
        #
        # self.vis_pose(best_pose, "_grasp")
        # grasp_pose = best_pose.pose
        # grasp_pose_stamped.pose = grasp_pose

        # === execute grasp (or stop) ===

        # # 1) if there is one left that is above threshold...
        # if best_pose:
        #     print("trying to grasp %s (%0.2f)" % (best_pose.name, best_pose.confidence))
        #
        #     # a) grasp
        #
        #     client_grasp = actionlib.SimpleActionClient('execute_grasp', ExecuteGraspAction)
        #     client_grasp.wait_for_server()
        #
        #     goal_grasp = ExecuteGraspGoal()
        #     goal_grasp.grasp_pose = grasp_pose_stamped
        #     goal_grasp.grasp_height = 0.03  # palm 3cm above grasp pose when grasping
        #     goal_grasp.safety_distance = 0.15  # approach from 15cm above
        #
        #     client_grasp.send_goal(goal_grasp)
        #     client_grasp.wait_for_result(rospy.Duration.from_sec(25.0))
        #     state = client_grasp.get_state()
        #     if state == 3:
        #         print
        #         client_grasp.get_result()
        #     elif state == 4:
        #         print
        #         'Goal aborted'
        #
        #     # b) handover routine
        #
        #     client_handover = actionlib.SimpleActionClient('handover', HandoverAction)
        #     client_handover.wait_for_server()
        #
        #     goal_handover = HandoverGoal()
        #     goal_handover.force_thresh = 0.5  # default is 0.2
        #
        #     client_handover.send_goal(goal_handover)
        #     client_handover.wait_for_result(rospy.Duration.from_sec(25.0))
        #     state = client_handover.get_state()
        #     if state == 3:
        #         print
        #         client_handover.get_result()
        #     elif state == 4:
        #         print
        #         'Goal aborted'
        #
        # # 2) ... none left -> stop
        # else:
        #     print("nothing left to grasp (or rejected)")

        self.working = False
        #return EmptyResponse()

        response.poses = poses
        return response

    def vis_detect(self, detections):
        width, height = self.rgb.width, self.rgb.height
        rgb = ros_numpy.numpify(self.rgb)

        vis_segmentation = rgb.copy()
        for detection in detections:
            name = detection.name

            bbox = detection.bbox
            obj_roi = [bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax]

            mask_ids = detection.mask
            mask_ids = np.array(mask_ids)
            obj_mask = np.zeros((height * width), dtype=np.uint8)
            obj_mask[mask_ids] = 1
            obj_mask = obj_mask.reshape((height, width))

            vis_segmentation[obj_mask > 0, 0] = 255
            vis_segmentation[obj_mask > 0, 1] = 0
            vis_segmentation[obj_mask > 0, 2] = 255

            cv2.rectangle(vis_segmentation, (obj_roi[1], obj_roi[0]), (obj_roi[3], obj_roi[2]), (255, 0, 255), 2)
            cv2.putText(vis_segmentation, name, (obj_roi[1], obj_roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, 1)
        vis_segmentation = np.uint8((np.float32(rgb) / 255 * 0.5 + np.float32(vis_segmentation) / 255 * 0.5) * 255)

        ros_vis_segmentation = ros_numpy.msgify(Image, vis_segmentation, encoding="rgb8")
        self.pub_segmentation.publish(ros_vis_segmentation)

    def vis_pose(self, poses):
        rgb = ros_numpy.numpify(self.rgb)
        camera_info = rospy.wait_for_message(CAMERA_INFO, CameraInfo, timeout=10)
        intrinsics = np.array(camera_info.K).reshape(3, 3)
        
        obj_ids = []
        obj_trafos = []
        for pose in poses:
           name = pose.name
           obj_id = -1
           for idx, obj_name in self.dataset.obj_names.items():
               if obj_name == name:
                   obj_id = int(idx)
                   break
           assert obj_id > 0  # should start from 1
           obj_ids.append(obj_id)

        #obj_ids = [obj_id]  # TODO expand to a list of poses
           T_obj = np.matrix(np.eye(4))
           T_obj[:3, :3] = scit.Rotation.from_quat(ros_numpy.numpify(pose.pose.orientation)).as_dcm()
           T_obj[:3, 3] = ros_numpy.numpify(pose.pose.position).reshape(3, 1)
           obj_trafos.append(T_obj)

        #obj_trafos = [T_obj]
        rendered = self.renderer.render(obj_ids, obj_trafos,
                                        np.matrix(np.eye(4)), intrinsics)#,
                                        #mode='color+depth+seg')

        vis_pose = np.float32(rgb.copy()) / 255
        highlight = np.float32(rendered[0]) / 255
        class_ids = np.unique(rendered[2])
        for class_id in class_ids:
            if class_id == 0:
                continue

            mask = rendered[2] == class_id
            vis_pose[mask] = highlight[mask] * 0.7 + vis_pose[mask] * 0.3

            _, contour, _ = cv2.findContours(np.uint8(mask), cv2.RETR_CCOMP,
                                             cv2.CHAIN_APPROX_TC89_L1)
            vis_pose = cv2.drawContours(vis_pose, contour, -1, (0, 1, 0), 1, lineType=cv2.LINE_AA)

        vis_pose = np.uint8(vis_pose * 255)

#        import matplotlib.pyplot as plt

#        plt.subplot(1, 3, 1)
#        plt.imshow(rgb)
#        plt.subplot(1, 3, 2)
#        plt.imshow(rendered[0])
#        plt.subplot(1, 3, 3)
#        plt.imshow(vis_pose)
#        plt.show()
        # TODO visualize detected plane

        ros_vis_initial = ros_numpy.msgify(Image, vis_pose, encoding="rgb8")
        self.pub_refined.publish(ros_vis_initial)

    def tf_pose(self, pose, suffix):
        self.pub_poses.sendTransform([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w], rospy.Time.now(), pose.name + suffix,
                                     self.rgb.header.frame_id)


if __name__ == "__main__":
    rospy.init_node("grasp_estimation_pipeline")

    grasper = Grasper()

    sub_rgb, sub_depth = message_filters.Subscriber(RGB_TOPIC, Image), message_filters.Subscriber(DEPTH_TOPIC, Image)
    rgbd = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth], 10, 0.3)
    rgbd.registerCallback(grasper.rgbd_callback)

    s = rospy.Service("get_poses", get_poses, grasper.grasp)
    print("Get poses ready.")

    rospy.spin()
