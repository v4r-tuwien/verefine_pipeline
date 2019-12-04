import numpy as np
import rospy
import ros_numpy
import actionlib

import message_filters
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import Image, Header
from std_srvs.srv import Empty, EmptyResponse
from object_detector_msgs.srv import detectron2_service_server, estimate_poses, refine_poses
from grasping_pipeline.msg import ExecuteGraspAction, ExecuteGraspGoal

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
            # refine candidate poses
            print("requesting pose refinement...")
            instance_poses = refine(self.rgb, self.depth, detection, instance_poses)
            print("   received refined poses.")

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
        grasp_pose_stamped.pose = grasp_pose

        # === execute grasp (or stop) ===
        # 1) if there is one left that is above threshold...
        if best_pose:
            print("trying to grasp %s (%0.2f)" % (best_pose.name, best_pose.confidence))

            # a) grasp

            client = actionlib.SimpleActionClient('execute_grasp', ExecuteGraspAction)
            client.wait_for_server()

            goal = ExecuteGraspGoal()
            goal.grasp_pose = grasp_pose_stamped
            goal.grasp_height = 0.03  # palm 3cm above grasp pose when grasping
            goal.safety_distance = 0.15  # approach from 15cm above

            client.send_goal(goal)
            client.wait_for_result(rospy.Duration.from_sec(25.0))
            state = client.get_state()
            if state == 3:
                print
                client.get_result()
            elif state == 4:
                print
                'Goal aborted'

            # b) handover routine

            # TODO handover service/action

        # 2) ... none left -> stop
        else:
            print("nothing left to grasp (or rejected)")

        self.working = False
        return EmptyResponse()


if __name__ == "__main__":
    rospy.init_node("grasp_estimation_pipeline")

    grasper = Grasper()

    sub_rgb, sub_depth = message_filters.Subscriber(RGB_TOPIC, Image), message_filters.Subscriber(DEPTH_TOPIC, Image)
    rgbd = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth], 10, 0.3)
    rgbd.registerCallback(grasper.rgbd_callback)

    s = rospy.Service("grasp_object", Empty, grasper.grasp)
    print("Object grasping ready.")

    rospy.spin()
