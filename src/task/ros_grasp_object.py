import numpy as np
import rospy
import ros_numpy

import message_filters
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse
from object_detector_msgs.srv import detectron2_service_server, estimate_poses, refine_poses


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

        # === annotate grasp ===

        # TODO simple top grasp -- add offset in object z-axis

        # # === execute grasp (or stop) ===
        # # 1) if there is one left that is above threshold...
        # if best_pose:
        #     print("trying to grasp %s (%0.2f)" % (best_pose.name, best_pose.confidence))
        #
        #     # a) grasp
        #
        #     # TODO grasp service/action
        #
        #     # b) handover routine
        #
        #     # TODO handover service/action
        #
        # # 2) ... none left -> stop
        # else:
        #     print("nothing left to grasp (or rejected)")
        #
        # return  # TODO return info on remaining objects

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
