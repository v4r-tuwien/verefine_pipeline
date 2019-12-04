import numpy as np
import rospy
import ros_numpy
import actionlib
import tf

import cv2

import message_filters
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import Image, Header
from std_srvs.srv import Empty, EmptyResponse
from object_detector_msgs.srv import detectron2_service_server, estimate_poses, refine_poses
from grasping_pipeline.action import ExecuteGraspAction, ExecuteGraspGoal
from sasha_handover.action import HandoverAction, HandoverGoal


# === define pipeline clients ===

def detect(rgb):
    rospy.wait_for_service('detect_objects')
    try:
        detect_objects = rospy.ServiceProxy('detect_objects', detectron2_service_server)
        response = detect_objects(rgb)
        return response.detections
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def estimate(rgb, depth, detection):
    rospy.wait_for_service('estimate_pose')
    try:
        estimate_pose = rospy.ServiceProxy('estimate_pose', estimate_poses)
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


class Grasper:

    def __init__(self):
        self.rgb = None
        self.depth = None
        self.working = False

        self.pub_segmentation = rospy.Publisher("/hsr-grasping/segmentation", Image)
        # self.pub_initial = rospy.Publisher("/hsr-grasping/initial_poses", Image)
        # self.pub_refined = rospy.Publisher("/hsr-grasping/refined_poses", Image)
        self.pub_poses = tf.TransformBroadcaster()

    def rgbd_callback(self, rgb, depth):
        print("callback...")
        if not self.working:
            print("   set images")
            self.rgb, self.depth = rgb, depth

    def grasp(self, req):

        # === check if we have an image ===

        if self.rgb is None or self.depth is None:
            print("no images available")
            return EmptyResponse()

        # === run pipeline ===
        self.working = True

        # detect all instances in image
        print("requesting detection...")
        detections = detect(self.rgb)
        print("   received detection.")
        self.vis_detect(detections.detections)

        # for each instance...
        poses = []
        confidences = []
        for detection in detections.detections:
            # reject based on detection score
            if detection.score < 0.1:
                print("detection of %s rejected" % detection.name)
                continue

            # estimate a set of candidate poses
            print("requesting pose estimate...")
            instance_poses = estimate(self.rgb, self.depth, detection)
            print("   received pose.")
            # for instance_pose in instance_poses:
            #   self.vis_pose(instance_poses, "_estimated")

            # refine candidate poses
            print("requesting pose refinement...")
            instance_poses = refine(self.rgb, self.depth, detection, instance_poses)
            print("   received refined poses.")
            # for instance_pose in instance_poses:
            #   self.vis_pose(instance_poses, "_refined")

            # reject by pose confidence
            print(",".join(["%0.3f" % pose.confidence for pose in instance_poses]))
            instance_poses = [pose for pose in instance_poses if pose.confidence > 0.3]
            # add to set of poses of detected instances
            if len(instance_poses) > 0:
                poses += instance_poses
                confidences += [pose.confidence for pose in instance_poses]
            else:
                print("all poses of %s rejected" % detection.name)
        assert len(poses) == len(confidences)

        # === select pose with highest confidence === TODO and scale by distance?

        if len(confidences) > 0:
            best_hypothesis = np.argmax(confidences)
            best_pose = poses[best_hypothesis]
            self.vis_pose(best_pose, "_pose")
            print("... now we would annotate pose for %s" % best_pose.name)
        else:
            best_pose = None
            print("no valid poses")
            return EmptyResponse()

        # === annotate grasp ===

        grasp_offsets = {
            "002_master_chef_can": [0, 0.065, 0],
            "003_cracker_box": [0, 0.100, 0],
            "004_sugar_box": [0, 0.083, 0],
            "005_tomato_soup_can": [0, 0.036, 0],
            "006_mustard_bottle": [0, 0.106, 0],  # slightly rotated around y
            "007_tuna_fish_can": [0, 0.008, 0],
            "008_pudding_box": [0, 0.015, 0],  # lying down, rotated by 45deg
            "009_gelatin_box": [0, 0.011, 0],  # lying down, slightly rotated
            "010_potted_meat_can": [0, 0.029, 0],
            "011_banana": [0, 0.019, 0],  # lying down, "Stil" in +z dir
            "019_pitcher_base": [0, 0.096, 0],
            "021_bleach_cleanser": [0, 0.137, 0],
            "024_bowl": [0.079, 0.027, 0],  # grasp point is on rim
            "025_mug": [0.047, 0.040, 0],  # grasp point is on rim
            "035_power_drill": [-0.024, 0.017, 0.041],  # lying down, grasp point is on handle
            "036_wood_block": [0, 0.085, 0],  # standing upright, slightly rotated
            "037_scissors": [-0.012, 0.005, -0.035],  # lying down, grasp point close to blades (base)
            "040_large_marker": [0, 0.009, 0],
            "051_large_clamp": [0, 0.013, 0],  # grasp point on joint, slightly rotated
            "052_extra_large_clamp": [0, 0.019, 0],  # grasp point on joint, rotated by 90deg
            "061_foam_brick": [0, 0.033, 0],  # holes up
        }
        T_grasp = np.matrix(np.eye(4))
        T_grasp[:3, 3] = np.squeeze(grasp_offsets[best_pose.name])

        # apply this as local transformation
        r, t = ros_numpy.numpify(best_pose.orientation), ros_numpy.numpify(best_pose.position)
        import scipy.spatial.transform as scit
        R = scit.Rotation.from_quat(r).as_dcm()

        T_obj = np.matrix(np.eye(4))
        T_obj[:3, :3] = R
        T_obj[:3, 3] = t

        T_grasp = T_obj * T_grasp
        print(T_grasp)

        r = scit.Rotation.from_dcm(T_grasp[:3, :3]).as_quat()
        t = T_grasp[:3, 3]
        grasp_pose_stamped = PoseStamped()
        grasp_pose_stamped.header = self.rgb.header

        grasp_pose = Pose()
        grasp_pose.position = ros_numpy.msgify(Point, t)
        grasp_pose.orientation = ros_numpy.msgify(Quaternion, r)
        self.vis_pose(grasp_pose, "_grasp")
        grasp_pose_stamped.pose = grasp_pose

        # === execute grasp (or stop) ===
        # 1) if there is one left that is above threshold...
        if best_pose:
            print("trying to grasp %s (%0.2f)" % (best_pose.name, best_pose.confidence))

            # a) grasp

            client_grasp = actionlib.SimpleActionClient('execute_grasp', ExecuteGraspAction)
            client_grasp.wait_for_server()

            goal_grasp = ExecuteGraspGoal()
            goal_grasp.grasp_pose = grasp_pose_stamped
            goal_grasp.grasp_height = 0.03  # palm 3cm above grasp pose when grasping
            goal_grasp.safety_distance = 0.15  # approach from 15cm above

            client_grasp.send_goal(goal_grasp)
            client_grasp.wait_for_result(rospy.Duration.from_sec(25.0))
            state = client_grasp.get_state()
            if state == 3:
                print
                client_grasp.get_result()
            elif state == 4:
                print
                'Goal aborted'

            # b) handover routine

            client_handover = actionlib.SimpleActionClient('handover', HandoverAction)
            client_handover.wait_for_server()

            goal_handover = HandoverGoal()
            goal_handover.force_thresh = 0.5  # default is 0.2

            client_handover.send_goal(goal_handover)
            client_handover.wait_for_result(rospy.Duration.from_sec(25.0))
            state = client_handover.get_state()
            if state == 3:
                print
                client_handover.get_result()
            elif state == 4:
                print
                'Goal aborted'

        # 2) ... none left -> stop
        else:
            print("nothing left to grasp (or rejected)")

        self.working = False
        return EmptyResponse()

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

            vis_segmentation[obj_mask, 0] = 255
            vis_segmentation[obj_mask, 1] = 0
            vis_segmentation[obj_mask, 2] = 255

            cv2.rectangle(vis_segmentation, (obj_roi[1], obj_roi[0]), (obj_roi[3], obj_roi[2]), (255, 0, 255), 2)
            cv2.putText(vis_segmentation, name, (obj_roi[1], obj_roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, 1)
        vis_segmentation = np.uint8((rgb / 255 * 0.5 + vis_segmentation / 255 * 0.5) * 255)

        ros_vis_segmentation = ros_numpy.msgify(vis_segmentation, Image)
        self.pub_segmentation.publish(ros_vis_segmentation)

    # TODO rendering-based visualizations for pose
    def vis_pose(self, pose, suffix):
        self.pub_poses.sendTransform(pose.position, pose.orientation, rospy.Time.now(), pose.name + suffix,
                                     self.rgb.header.frame_id)

if __name__ == "__main__":
    rospy.init_node("grasp_estimation_pipeline")

    grasper = Grasper()

    sub_rgb, sub_depth = message_filters.Subscriber(RGB_TOPIC, Image), message_filters.Subscriber(DEPTH_TOPIC, Image)
    rgbd = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth], 10, 0.3)
    rgbd.registerCallback(grasper.rgbd_callback)

    s = rospy.Service("grasp_object", Empty, grasper.grasp)
    print("Object grasping ready.")

    rospy.spin()
