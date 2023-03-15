import rospy
import ros_numpy
from actionlib import SimpleActionClient
from tracebot_msgs.msg import LocateObjectAction, LocateObjectGoal, LocateObjectResult
import numpy as np
from scipy.spatial.transform.rotation import Rotation


def ros_pose_to_mat(pose):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Rotation.from_quat(ros_numpy.numpify(pose.orientation)).as_matrix()
    mat[:3, 3] = ros_numpy.numpify(pose.position)
    return mat


if __name__ == '__main__':
    rospy.init_node('dummy_client')
    # run action
    client = SimpleActionClient('/locate_object', LocateObjectAction)
    client.wait_for_server()
    goal = LocateObjectGoal(object_to_locate='Canister')
    client.send_goal(goal)
    client.wait_for_result()
    result = client.get_result()
    # print results
    print(f"Results for image {result.color_image.header.seq % 3 + 1}:\n===")
    print("\n---\n".join([f"{obj_type} (conf={conf:0.3f}):\n{ros_pose_to_mat(pose)}" for obj_type, conf, pose
                          in zip(result.object_types, result.confidences, result.object_poses)]))
