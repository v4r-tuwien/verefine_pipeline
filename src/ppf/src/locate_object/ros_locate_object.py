# --- ROS

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

#import message_filters
import rospy
import ros_numpy
import tf2_ros

from actionlib import SimpleActionServer
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
#from tracebot_msgs.msg import LocateObjectAction, LocateObjectResult
from object_detector_msgs.msg import BoundingBox, Detection, PoseWithConfidence
from object_detector_msgs.srv import estimate_poses, estimate_posesResponse
# --- Segmentation + Pose Estimation
import copy
import math
import numpy as np
import open3d as o3d
from scipy.spatial.transform.rotation import Rotation
import os
import yaml
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ppf", "build"))
from pyppf import PPF, ObjectHypothesis
from plane_detector import PlaneDetector
import json
from util.dataset import YcbvDataset

# --- Pose Refinement + Verification
import trimesh
import cv2 as cv
import sys
sys.path.append("/ppf/src/")
sys.path.append("/ppf/src/verefine")
from verefine import config
import verefine.p2pl as refiner
from verefine.renderer import Renderer
from verefine.simulator import Simulator
from verefine.verefine import Verefine

# --- debug
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def make_dir(path):
    # Catch all errors except the one raised by the folder already existing
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def get_cont_sympose(rot_pose, sym):
    """
        Orient an object with a continuous symmety axis consistently towards
        the camera
    Args
        rot_pose: pose of the object in the camera frame
        sym: 3 value vector, [xyz], 1 for the axis of symmetry, 0 otherwise
    """

    cam_in_obj = np.dot(np.linalg.inv(rot_pose), (0, 0, 0, 1))
    if sym[0][2] == 1:
        alpha = math.atan2(cam_in_obj[1], cam_in_obj[0])
        rota = np.dot(rot_pose[:3, :3],
                      Rotation.from_euler('xyz', [0.0, 0.0, alpha]).as_matrix())
    elif sym[0][1] == 1:
        alpha = math.atan2(cam_in_obj[0], cam_in_obj[2])
        rota = np.dot(rot_pose[:3, :3],
                      Rotation.from_euler('xyz', [0.0, alpha, 0.0]).as_matrix())
    elif sym[0][0] == 1:
        alpha = math.atan2(cam_in_obj[2], cam_in_obj[1])
        rota = np.dot(rot_pose[:3, :3],
                      Rotation.from_euler('xyz', [alpha, 0.0, 0.0]).as_matrix())

    return rota


class LocateObject:

    def __init__(self, name):
        # custom parameters
        self.down_scale = rospy.get_param('/locateobject/down_scale', 1)  # down_scale speeds things up by already subsampling the image [power of 2]
        self.max_dist = rospy.get_param('/locateobject/max_dist', 1.0)  # crop points farther than this [meters]
        self.slop = 0.2  # max delay between rgb and depth image [seconds]
        self.use_refinement = rospy.get_param('/locateobject/use_refinement', True)
        self.use_verification = rospy.get_param('/locateobject/use_verification', True)
        self.use_verefine = True and self.use_refinement and self.use_verification  # combines both
        self.debug_visualization = rospy.get_param('/locateobject/debug_visualization', True)  # visualize the process on an image topic
        if self.debug_visualization:
            self.verefine_debub_img_pub = rospy.Publisher("/pose_estimator/ppf_verefine_result", Image, queue_size=1)

        self.simulator = None
        # camera
        self.rgb, self.depth = None, None
        #self.color_topic = rospy.get_param('/locateobject/color_topic')
        #self.depth_topic = rospy.get_param('/locateobject/depth_topic')
        #self.camera_info_topic = rospy.get_param('/locateobject/camera_info_topic')

                # dataset infos
        cfg_dir = rospy.get_param('/pose_estimator/cfg_dir')
        models_dir = rospy.get_param('/pose_estimator/models_dir')
        #self.object_models = ["Canister", "DrainTray"]  # empty: use all models
        
        #object_to_m = [1, 1e-3]
        #object_to_mm = [scale * 1e3 for scale in object_to_m]

        dataset = YcbvDataset()        
        self.im_width = rospy.get_param('/pose_estimator/im_width')
        self.im_height = rospy.get_param('/pose_estimator/im_height')
        self.im_K = np.asarray(rospy.get_param('/pose_estimator/intrinsics'))
        self.ycbv_names_json = rospy.get_param('/pose_estimator/ycbv_names')
        self.ycbv_verefine= rospy.get_param('/pose_estimator/ycbv_verefine')
        meta_vf = json.load(open(self.ycbv_verefine, 'r'))
        self.obj_meta = meta_vf['objects']
        self.obj_ids = list(self.obj_meta.keys())
        object_to_m = np.ones(len(self.obj_ids))
        object_to_mm = [scale * 1e3 for scale in object_to_m]
        names = [self.obj_meta[id]['name'] for id in self.obj_meta.keys()]
        self.object_models = names 

        # get meshes and point clouds (sampled from mesh)
        if self.use_refinement or self.use_verification:
            # TODO create centered models of same dimension

            mesh_paths = [f"{models_dir}/{obj_model}.stl" for obj_model in self.object_models]
            cloud_paths = [f"{models_dir}/{obj_model}/3D_model.pcd"  for obj_model in self.object_models]
            meshes = dataset.meshes
            self.meshes_sampled = []
            for mesh, to_m, cloud_path in zip(meshes, object_to_m, cloud_paths):
                if os.path.exists(cloud_path):
                    pcd = o3d.io.read_point_cloud(cloud_path)
                else:
                    # Create the folder that will contain the generated point cloud
                    make_dir(os.path.abspath(os.path.join(cloud_path, os.pardir)))

                    # Generate the point cloud
                    points, face_indices = trimesh.sample.sample_surface_even(mesh, 4096)
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points * to_m))
                    pcd.normals = o3d.utility.Vector3dVector(mesh.face_normals[face_indices])
                    o3d.io.write_point_cloud(cloud_path, pcd)
                self.meshes_sampled.append(pcd)
            meshes = [mesh.apply_scale(to_mm) for mesh, to_mm in zip(meshes, object_to_mm)]  # for renderer

        # create plane detector and PPF
        # pd_roi_data = yaml.load(open("/canister/src/locate_object/plane_detector_roi.yml", 'r'), Loader=yaml.FullLoader)
        if rospy.get_param('/locateobject/use_roi'):
            roi_center = np.array(rospy.get_param('/locateobject/roi_center'))
            roi_rotation = np.array(rospy.get_param('/locateobject/roi_rotation')).reshape((3, 3))
            roi_extent = np.array(rospy.get_param('/locateobject/roi_extent'))
            roi_bbox = o3d.geometry.OrientedBoundingBox(roi_center, roi_rotation, roi_extent)
        else:
            roi_bbox = None

        pd_to_meters = rospy.get_param('/locateobject/to_meters', 1e-3)
        distance_threshold = rospy.get_param('/locateobject/distance_threshold', 0.01)

        self.detector = PlaneDetector(self.im_width, self.im_height, self.im_K,
                                      self.down_scale, roi_bbox=roi_bbox,
                                      to_meters=pd_to_meters, distance_threshold=distance_threshold)
        
        self.ppf = PPF(cfg_dir, models_dir, self.object_models)

        # prepare renderer for verification
        if self.use_verification:
            self.renderer = Renderer(meshes, self.im_width, self.im_height)

        # prepare verefine
        if self.use_verefine:
            # TODO create colliders from decomposition, get CoM (or center at CoM)
            self.simulator = Simulator(self.object_models, mesh_paths,
                                       [[0, 0, 0]]*len(mesh_paths), object_to_m)
            self.verefine = Verefine(self.object_models, self.meshes_sampled, refiner, self.simulator, self.renderer)
        #sub_rgb, sub_depth = message_filters.Subscriber(self.color_topic, Image),\
        #                     message_filters.Subscriber(self.depth_topic, Image)
        #sub_rgbd = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth], 10, self.slop)
        #sub_rgbd.registerCallback(self._update_image)
        #rospy.loginfo(f"[{name}] Waiting for camera info...")
        #self.camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
        #self.im_K = np.array([v for v in self.camera_info.K]).reshape(3, 3)
        #self.im_width, self.im_height = self.camera_info.width, self.camera_info.height

        self._server = rospy.Service("estimate_poses", estimate_poses, self.execute_cb)
        rospy.loginfo("[{}] Service ready".format(name))

        if rospy.get_param('/locateobject/publish_tf'):
            self._br = tf2_ros.TransformBroadcaster()
            self._publish_tf()

    #def _update_image(self, rgb, depth):
    #    self.rgb, self.depth = rgb, depth

    def _publish_tf(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if not hasattr(self, '_last_result'):
                rate.sleep()
                continue

            object_types_count = {}
            for idx, otype in enumerate(self._last_result.object_types):
                ocount = object_types_count.get(otype, 1)
                object_types_count[otype] = ocount + 1
                opose = self._last_result.object_poses[idx]

                # Create the TransformStamped message
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self.camera_info.header.frame_id
                t.child_frame_id = "{}_{}".format(otype, ocount)
                t.transform.translation.x = opose.position.x
                t.transform.translation.y = opose.position.y
                t.transform.translation.z = opose.position.z
                t.transform.rotation.x = opose.orientation.x
                t.transform.rotation.y = opose.orientation.y
                t.transform.rotation.z = opose.orientation.z
                t.transform.rotation.w = opose.orientation.w

                self._br.sendTransform(t)

            rate.sleep()


    def execute_cb(self, goal):

        self.rgb = goal.rgb
        self.depth = goal.depth


        # create server
        #self._server = SimpleActionServer(name, LocateObjectAction, execute_cb=self.execute_cb, auto_start=False)
        #self._server.start()

        if self.rgb is None or self.depth is None:
            #self._server.set_aborted(text=f"No camera image available")
            return None
        #elif goal.object_to_locate not in self.object_models + [""]:
            #self._server.set_aborted(text=f"Unknown object_to_locate; "
            #                             f"Available objects are: {self.object_models}.")
            #return None
        else:
            # == get images and objects to search
            rgb, depth = ros_numpy.numpify(self.rgb), ros_numpy.numpify(self.depth)
            obj_models_to_search = [goal.det.name] if goal.det.name != "" else []

            if self.use_verification:
                # -- compute normal image
                D_px = depth.copy()
                # inpaint missing depth values
                D_px = cv.inpaint(D_px.astype(np.float32), np.uint8(D_px == 0), 3, cv.INPAINT_NS)
                # blur
                blur_size = (7, 7)
                D_px = cv.GaussianBlur(D_px, blur_size, sigmaX=10.0)
                # get derivatives
                kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                dzdx = cv.filter2D(D_px, -1, kernelx)
                dzdy = cv.filter2D(D_px, -1, kernely)
                # gradient ~ normal
                normal = np.dstack((dzdy, dzdx, D_px != 0.0))  # only where we have a depth value
                n = np.linalg.norm(normal, axis=2)
                n = np.dstack((n, n, n))
                normal = np.divide(normal, n, where=(n != 0))
                # remove invalid values
                normal[n == 0] = 0.0
                normal[depth == 0] = 0.0

            # == preprocessing
            # -- get plane pose, inliers and scene points
            plane_pose, plane_pcd, scene_pcd, cloud_pcd, plane_indices = self.detector.detect(rgb, depth, self.max_dist)
            plane_pcd = plane_pcd.transform(plane_pose)
            scene_pcd = scene_pcd.transform(plane_pose)

            # -- plane pop-out
            # outlier removal
            plane_pcd, _ = plane_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.5)
            scene_pcd, _ = scene_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
            # plane pop-out: select points within the bounds and above the xy-plane
            plane_bbox = np.asarray(plane_pcd.get_axis_aligned_bounding_box().get_box_points())
            plane_bbox[3:7, 2] = scene_pcd.get_max_bound()[2]
            plane_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(plane_bbox))
            object_pcd = scene_pcd.crop(plane_bbox)

            # == compute object pose using PPF
            # -- prepare input: transform back to camera coordinates and create array in required format
            object_pcd = object_pcd.transform(np.linalg.inv(plane_pose))
            object_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
            object_pcd.orient_normals_to_align_with_direction([0, 0, -1])
            scene_points = np.hstack([np.asarray(object_pcd.points), np.asarray(object_pcd.colors),
                                      np.asarray(object_pcd.normals)]).astype(np.float64)


            # -- detect and initial estimate poses
            # note: generates at most PPFRecognitionPipelineParameter.correspondences_per_scene_point_ per object (defaults to 3)
            # note: returns at most PPFRecognitionPipelineParameter.max_hypotheses_ per object (defaults to 3)
            objects_hypotheses = self.ppf.estimate(points_array=scene_points, obj_models_to_search=obj_models_to_search)

            if self.use_refinement and not self.use_verefine:
                # == refine all poses using ICP
                for oi, object_hypotheses in enumerate(objects_hypotheses):
                    for hi, object_hypothesis in enumerate(object_hypotheses):
                        if self.debug_visualization:
                            obj_idx = self.object_models.index(object_hypothesis.model_id)
                            # visualization
                            D_ren, N_ren = self.renderer.render([obj_idx], [object_hypothesis.transform],
                                                                np.eye(4), self.im_K)
                            mask_before = D_ren > 0

                        # refine
                        obj_idx = self.object_models.index(object_hypothesis.model_id)
                        T_ref = refiner.refine(object_hypothesis.transform, object_pcd, self.meshes_sampled[obj_idx],
                                               p_distance=config.ICP_P_DISTANCE, iterations=config.ICP_ITERATIONS)
                        # update pose estimate
                        object_hypothesis.transform = T_ref

                        if self.debug_visualization:
                            # visualization
                            vis = rgb.copy()
                            D_ren, N_ren = self.renderer.render([obj_idx], [object_hypothesis.transform],
                                                                np.eye(4), self.im_K)
                            mask_after = D_ren > 0
                            # overlay outlines
                            contour, _ = cv.findContours(np.uint8(mask_before), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
                            vis = cv.drawContours(vis, contour, -1, (255, 0, 220), 2, lineType=cv.LINE_AA)
                            contour, _ = cv.findContours(np.uint8(mask_after), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
                            vis = cv.drawContours(vis, contour, -1, (0, 220, 255), 2, lineType=cv.LINE_AA)
                            self.verefine_debub_img_pub.publish(ros_numpy.msgify(Image, vis, encoding="rgb8"))

            if self.use_verification and not self.use_verefine:
                # == select best pose per object using rendering-based verification
                self.renderer.set_observation(depth, normal)
                new_objs_hs = []
                for oi, object_hypotheses in enumerate(objects_hypotheses):
                    best_score, best_hi = -1, -1
                    for hi, object_hypothesis in enumerate(object_hypotheses):
                        # score hypothesis
                        obj_idx = self.object_models.index(object_hypothesis.model_id)
                        if self.debug_visualization:
                            score, f_d, f_n = self.renderer.compute_score([obj_idx], [object_hypothesis.transform],
                                                                          np.eye(4), self.im_K, return_map=True)
                            # visualization
                            vis = rgb.copy()
                            D_ren, N_ren = self.renderer.render([obj_idx], [object_hypothesis.transform],
                                                                np.eye(4), self.im_K)
                            mask = D_ren > 0
                            # overlay score
                            vis_score = (f_d*0.5 + f_n*0.5)
                            colormap = cm.inferno
                            normalize = mcolors.Normalize(vmin=0.0, vmax=1.0)
                            s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
                            vis_score = np.uint8(s_map.to_rgba(vis_score)[..., :3] * 255)
                            vis[mask] = vis_score[mask]
                            # overlay outline
                            contour, _ = cv.findContours(np.uint8(mask), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
                            vis = cv.drawContours(vis, contour, -1, (0, 220, 255), 2, lineType=cv.LINE_AA)
                            self.verefine_debub_img_pub.publish(ros_numpy.msgify(Image, vis, encoding="rgb8"))
                        else:
                            score = self.renderer.compute_score([obj_idx], [object_hypothesis.transform],
                                                                np.eye(4), self.im_K)

                        if score > best_score:
                            best_score = score
                            best_hi = hi
                        object_hypothesis.confidence = score
                    # keep only best per object
                    new_objs_hs.append([object_hypotheses[best_hi]])
                objects_hypotheses = new_objs_hs

            if self.use_verefine:
                # == refine and select best pose per object using verefine and only return those

                if self.debug_visualization:
                    masks_before = []
                    for oi, object_hypotheses in enumerate(objects_hypotheses):
                        object_masks_before = []
                        for hi, object_hypothesis in enumerate(object_hypotheses):
                            if self.debug_visualization:
                                obj_idx = self.object_models.index(object_hypothesis.model_id)
                                # visualization
                                D_ren, N_ren = self.renderer.render([obj_idx], [object_hypothesis.transform],
                                                                    np.eye(4), self.im_K)
                                mask_before = D_ren > 0
                                object_masks_before.append(mask_before)
                        masks_before.append(object_masks_before)

                # -- to verefine format
                observation = {
                    'depth': depth,
                    'normal': normal,
                    'dependencies': None,
                    'extrinsics': plane_pose,
                    'intrinsics': self.im_K
                }
                hs = []
                for oi, object_hypotheses in enumerate(objects_hypotheses):
                    obj_hs = []
                    for hi, object_hypothesis in enumerate(object_hypotheses):
                        hypothesis = {
                            'obj_id': object_hypothesis.model_id,
                            'pose': object_hypothesis.transform,
                            'confidence': object_hypothesis.confidence,
                            'cloud_obs': object_pcd
                        }
                        obj_hs.append(hypothesis)
                    hs.append(obj_hs)

                # -- run refinement with verification - returns best per object
                hs = self.verefine.refine(observation, hs)

                # -- to original (PPF-style) format
                objects_hypotheses = []
                for oi, h in enumerate(hs):
                    object_hypothesis = ObjectHypothesis()
                    object_hypothesis.model_id = h['obj_id']
                    object_hypothesis.transform = h['pose']
                    object_hypothesis.confidence = h['confidence']
                    objects_hypotheses.append([object_hypothesis])

                    if self.debug_visualization:
                        obj_idx = self.object_models.index(object_hypothesis.model_id)
                        score, f_d, f_n = self.renderer.compute_score([obj_idx], [object_hypothesis.transform],
                                                                      np.eye(4), self.im_K, return_map=True)
                        print(f_d)
                        print(f_n)
                        # visualization
                        vis = rgb.copy()
                        D_ren, N_ren = self.renderer.render([obj_idx], [object_hypothesis.transform],
                                                            np.eye(4), self.im_K)
                        mask = D_ren > 0
                        # overlay score
                        vis_score = (f_d * 0.5 + f_n * 0.5)
                        colormap = cm.inferno
                        normalize = mcolors.Normalize(vmin=0.0, vmax=1.0)
                        s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
                        vis_score = np.uint8(s_map.to_rgba(vis_score)[..., :3] * 255)
                        vis[mask] = np.uint8(vis[mask] * 0.7 + vis_score[mask] * 0.3)
                        # overlay outline
                        vis_wo_contours = vis.copy()
                        for mask_before in masks_before[oi]:
                            contour, _ = cv.findContours(np.uint8(mask_before), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
                            vis = cv.drawContours(vis, contour, -1, (255, 0, 220), 2, lineType=cv.LINE_AA)
                        contour, _ = cv.findContours(np.uint8(mask), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
                        vis = cv.drawContours(vis, contour, -1, (0, 220, 255), 2, lineType=cv.LINE_AA)
                        vis = np.uint8(vis_wo_contours * 0.7 + vis * 0.3)
                        self.verefine_debub_img_pub.publish(ros_numpy.msgify(Image, vis, encoding="rgb8"))

            # == parse and return poses
            object_poses, object_types, confidences = [], [], []
            for oi, object_hypotheses in enumerate(objects_hypotheses):
                print("== Object {}".format(oi))
                for hi, object_hypothesis in enumerate(object_hypotheses):
                    print("-- Hypothesis {}".format(hi))
                    print("   model id: {}".format(object_hypothesis.model_id))
                    print("   transform:")
                    print(object_hypothesis.transform)
                    print("   confidence: {}".format(object_hypothesis.confidence))

                    object_pose = Pose()
                    object_pose.position = ros_numpy.msgify(Point, object_hypothesis.transform[:3, 3])
                    obj_rot = object_hypothesis.transform[:3, :3]

                    if object_hypothesis.model_id == 'Canister':
                        obj_rot = get_cont_sympose(object_hypothesis.transform, [[0, 0, 1]])

                    object_pose.orientation = ros_numpy.msgify(Quaternion,
                                                               Rotation.from_matrix(obj_rot).as_quat())
                    object_poses.append(object_pose)
                    object_types.append(object_hypothesis.model_id)
                    confidences.append(object_hypothesis.confidence)

            # result = LocateObjectResult()
            # result.header = self.rgb.header
            # result.color_image = self.rgb
            # result.depth_image = self.depth
            # result.camera_info = self.camera_info
            # result.object_poses = object_poses
            # result.object_types = object_types
            # result.confidences = confidences

            result = estimate_posesResponse()
            result.poses = []
            for index, pose in enumerate(object_poses):
                poseWithConfidence = PoseWithConfidence()
                poseWithConfidence.name = object_types[index]
                # for score compare with densefusion estimator
                if confidences[index] * 35 <= 1:
                    poseWithConfidence.confidence = confidences[index] * 35
                else:
                    poseWithConfidence.confidence = 1
                poseWithConfidence.pose = pose
                result.poses.append(poseWithConfidence)

            return (result)


if __name__ == '__main__':
    rospy.init_node('ppf_estimation_and_verefine_refinement')
    print(rospy.get_name())
    server = LocateObject(rospy.get_name())
    rospy.spin()
