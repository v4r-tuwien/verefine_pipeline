# Note: Adapted from Kiru's ICP implementation.

import numpy as np
import cv2
from scipy import ndimage

from src.verefine.refiner_interface import Refiner

from src.util.renderer import Renderer


class Icp(Refiner):

    def __init__(self, dataset):
        Refiner.__init__(self)
        self.renderer = Renderer(dataset)

    def getXYZ(self, depth, fx, fy, cx, cy, bbox=np.array([0])):
        # get x,y,z coordinate in mm dimension
        uv_table = np.zeros((depth.shape[0], depth.shape[1], 2), dtype=np.int16)
        column = np.arange(0, depth.shape[0])
        uv_table[:, :, 1] = np.arange(0, depth.shape[1]) - cx  # x-c_x (u)
        uv_table[:, :, 0] = column[:, np.newaxis] - cy  # y-c_y (v)

        if (bbox.shape[0] == 1):
            xyz = np.zeros((depth.shape[0], depth.shape[1], 3))  # x,y,z
            xyz[:, :, 0] = uv_table[:, :, 1] * depth * 1 / fx
            xyz[:, :, 1] = uv_table[:, :, 0] * depth * 1 / fy
            xyz[:, :, 2] = depth
        else:  # when boundry region is given
            xyz = np.zeros((bbox[2] - bbox[0], bbox[3] - bbox[1], 3))  # x,y,z
            xyz[:, :, 0] = uv_table[bbox[0]:bbox[2], bbox[1]:bbox[3], 1] * depth[bbox[0]:bbox[2], bbox[1]:bbox[3]] * 1 / fx
            xyz[:, :, 1] = uv_table[bbox[0]:bbox[2], bbox[1]:bbox[3], 0] * depth[bbox[0]:bbox[2], bbox[1]:bbox[3]] * 1 / fy
            xyz[:, :, 2] = depth[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        return xyz


    def get_normal(self, depth_refine, fx=-1, fy=-1, cx=-1, cy=-1, bbox=np.array([0]), refine=True):
        '''
        fast normal computation
        '''
        res_y = depth_refine.shape[0]
        res_x = depth_refine.shape[1]
        centerX = cx
        centerY = cy
        constant_x = 1 / fx
        constant_y = 1 / fy

        if (refine):
            depth_refine = np.nan_to_num(depth_refine)
            mask = np.zeros_like(depth_refine).astype(np.uint8)
            mask[depth_refine == 0] = 1
            depth_refine = depth_refine.astype(np.float32)
            depth_refine = cv2.inpaint(depth_refine, mask, 2, cv2.INPAINT_NS)
            depth_refine = depth_refine.astype(np.float)
            depth_refine = ndimage.gaussian_filter(depth_refine, 2)

        uv_table = np.zeros((res_y, res_x, 2), dtype=np.int16)
        column = np.arange(0, res_y)
        uv_table[:, :, 1] = np.arange(0, res_x) - centerX  # x-c_x (u)
        uv_table[:, :, 0] = column[:, np.newaxis] - centerY  # y-c_y (v)

        if (bbox.shape[0] == 4):
            uv_table = uv_table[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            v_x = np.zeros((bbox[2] - bbox[0], bbox[3] - bbox[1], 3))
            v_y = np.zeros((bbox[2] - bbox[0], bbox[3] - bbox[1], 3))
            normals = np.zeros((bbox[2] - bbox[0], bbox[3] - bbox[1], 3))
            depth_refine = depth_refine[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        else:
            v_x = np.zeros((res_y, res_x, 3))
            v_y = np.zeros((res_y, res_x, 3))
            normals = np.zeros((res_y, res_x, 3))

        uv_table_sign = np.copy(uv_table)
        uv_table = np.abs(np.copy(uv_table))

        dig = np.gradient(depth_refine, 2, edge_order=2)
        v_y[:, :, 0] = uv_table_sign[:, :, 1] * constant_x * dig[0]
        v_y[:, :, 1] = depth_refine * constant_y + (uv_table_sign[:, :, 0] * constant_y) * dig[0]
        v_y[:, :, 2] = dig[0]

        v_x[:, :, 0] = depth_refine * constant_x + uv_table_sign[:, :, 1] * constant_x * dig[1]
        v_x[:, :, 1] = uv_table_sign[:, :, 0] * constant_y * dig[1]
        v_x[:, :, 2] = dig[1]

        cross = np.cross(v_x.reshape(-1, 3), v_y.reshape(-1, 3))
        norm = np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)
        norm[norm == 0] = 1
        cross = cross / norm
        if (bbox.shape[0] == 4):
            cross = cross.reshape((bbox[2] - bbox[0], bbox[3] - bbox[1], 3))
        else:
            cross = cross.reshape(res_y, res_x, 3)
        cross = np.nan_to_num(cross)
        return cross

    def get_bbox_from_mask(self, mask):
        vu = np.where(mask)
        if(len(vu[0])>0):
            return np.array([np.min(vu[0]),np.min(vu[1]),np.max(vu[0]),np.max(vu[1])],np.int)
        else:
            return np.zeros((4),np.int)

    def icp_refinement(self, pts_tgt, obj_model, rot_pred, tra_pred, cam_K, ren):
        centroid_tgt = np.array([np.mean(pts_tgt[:, 0]), np.mean(pts_tgt[:, 1]), np.mean(pts_tgt[:, 2])])
        if (tra_pred[2] < 300 or tra_pred[2] > 5000):
            tra_pred = centroid_tgt * 1000

        img_init, depth_init = self.renderer.render(obj_model, rot_pred, tra_pred / 1000, cam_K, ren)
        init_mask = depth_init > 0
        bbox_init = self.get_bbox_from_mask(init_mask > 0)
        tf = np.eye(4)
        if (bbox_init[2] - bbox_init[0] < 10 or bbox_init[3] - bbox_init[1] < 10):
            return tf, -1
        if (np.sum(init_mask) < 10):
            return tf, -1
        points_src = np.zeros((bbox_init[2] - bbox_init[0], bbox_init[3] - bbox_init[1], 6), np.float32)
        points_src[:, :, :3] = self.getXYZ(depth_init, cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2], bbox_init)
        points_src[:, :, 3:] = self.get_normal(depth_init, fx=cam_K[0, 0], fy=cam_K[1, 1], cx=cam_K[0, 2], cy=cam_K[1, 2],
                                          refine=True, bbox=bbox_init)
        points_src = points_src[init_mask[bbox_init[0]:bbox_init[2], bbox_init[1]:bbox_init[3]] > 0]

        # adjust the initial translation using centroids of visible points
        centroid_src = np.array([np.mean(points_src[:, 0]), np.mean(points_src[:, 1]), np.mean(points_src[:, 2])])
        trans_adjust = centroid_tgt - centroid_src
        tra_pred = tra_pred + trans_adjust * 1000
        points_src[:, :3] += trans_adjust

        icp_fnc = cv2.ppf_match_3d_ICP(100, tolerence=0.05, numLevels=4)  # 1cm
        retval, residual, pose = icp_fnc.registerModelToScene(points_src.reshape(-1, 6), pts_tgt.reshape(-1, 6))

        tf[:3, :3] = rot_pred
        tf[:3, 3] = tra_pred / 1000  # in m
        tf = np.matmul(pose, tf)
        return tf, residual


import pcl
import open3d as o3d
import sys
sys.path.append("/home/dominik/projects/hsr-grasping/src/icp/cpp/build")
import icp
from scipy.spatial.transform.rotation import Rotation


class TrimmedIcp(Refiner):

    def __init__(self, renderer, intrinsics, dataset, mode):
        Refiner.__init__(self, intrinsics, dataset, mode=mode)

        self.renderer = renderer
        self.width, self.height = renderer.width, renderer.height
        self.umap = np.array([[j for _ in range(self.width)] for j in range(self.height)])
        self.vmap = np.array([[i for i in range(self.width)] for _ in range(self.height)])
        self.num_samples = 1000  # TODO how many samples required?

    def depth_to_cloud(self, depth, intrinsics, label=None, roi=None):

        mask_depth = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0))

        if label is not None:
            # get mask (from segmentation + valid depth values)
            mask_label = np.ma.getmaskarray(np.ma.masked_equal(label, 1))
            mask = mask_label * mask_depth
        else:
            mask = mask_depth

        # get samples in mask
        choose = mask.flatten().nonzero()[0]
        if self.num_samples > 0:
            if len(choose) > self.num_samples:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_samples] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            elif len(choose) == 0:
                return np.array([])
            else:
                choose = np.pad(choose, (0, self.num_samples - len(choose)), 'wrap')

        # point cloud from depth -- image space (u, v, D) to camera space (X, Y, D)
        D_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        u_masked = self.umap.flatten()[choose][:, np.newaxis].astype(np.float32)
        v_masked = self.vmap.flatten()[choose][:, np.newaxis].astype(np.float32)

        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        D_masked = D_masked  # in meters
        X_masked = (v_masked - cx) * D_masked / fx
        Y_masked = (u_masked - cy) * D_masked / fy
        cloud = np.concatenate((X_masked, Y_masked, D_masked), axis=1)

        return cloud

    def refine(self, rgb, depth, intrinsics, roi, mask, obj_id,
               estimate, iterations, cloud_obs=None, cloud_ren=None):

        if cloud_obs is None:
            cloud_obs = self.depth_to_cloud(depth / 1000.0, intrinsics, mask, roi)
        if cloud_obs.shape[0] == 0:
            return estimate

        # TODO replace estimate with hypothesis -- then we can just call render method of h
        # create estimated point cloud (TODO could just load model point cloud once and transform it here)
        obj_id = self.renderer.dataset.objlist.index(obj_id)

        q, t, c = estimate
        obj_T = np.matrix(np.eye(4))
        obj_T[:3, :3] = Rotation.from_quat(q).as_dcm()
        obj_T[:3, 3] = t.reshape(3, 1)
        rendered = self.renderer.render([obj_id], [obj_T],
                                   np.matrix(np.eye(4)), intrinsics,
                                   mode='depth')
        cloud_ren = self.depth_to_cloud(rendered[1], intrinsics, rendered[1] > 0)
        if cloud_ren.shape[0] == 0:
            return estimate

        # # -- icp
        # # a) using pcl bindings
        # # to pcd
        # cloud_in = pcl.PointCloud()
        # cloud_out = pcl.PointCloud()
        # cloud_in.from_array(cloud_obs)
        # cloud_out.from_array(cloud_ren)
        #
        # # icp
        # icp = cloud_in.make_IterativeClosestPoint()
        # converged, T, _, fit = icp.icp(cloud_in, cloud_out)

        # # b) using o3d
        # cloud_in = o3d.PointCloud()
        # cloud_out = o3d.PointCloud()
        # cloud_in.points = o3d.Vector3dVector(cloud_obs)
        # cloud_out.points = o3d.Vector3dVector(cloud_ren)
        #
        # reg_p2p = o3d.registration.registration_icp(
        #     cloud_in, cloud_out, 0.01, np.array(np.eye(4)),
        #     o3d.registration.TransformationEstimationPointToPoint(),
        #     o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30)
        # )
        # T = reg_p2p.transformation

        # c) using own python bindings to pcl
        T = icp.tricp(cloud_ren, cloud_obs, 0.5)
        # TODO icp.icp (would this require normals? then we'd have to adapt the cpp code)

        # -- apply trafo
        new_obj_T = np.matrix(T) * obj_T

        # # TODO debug
        # new_rendered = self.renderer.render([obj_id], [new_obj_T],
        #                                     np.matrix(np.eye(4)), intrinsics,
        #                                     mode='depth')
        # import matplotlib.pyplot as plt
        # plt.subplot(1, 3, 1)
        # plt.imshow(mask)
        # plt.subplot(1, 3, 2)
        # plt.imshow(rendered[1])
        # plt.subplot(1, 3, 3)
        # plt.imshow(new_rendered[1])
        # plt.show()

        return Rotation.from_dcm(new_obj_T[:3, :3]).as_quat(), np.array(new_obj_T[:3, 3]).T[0], c
