import numpy as np
from skimage.transform import resize
import pcl


class PlaneDetector:
    """
    Based on PCL tutorial: http://pointclouds.org/documentation/tutorials/planar_segmentation.php
    """

    def __init__(self, width, height, intrinsics, down_scale=1):
        self.down_scale = down_scale
        self.down_width, self.down_height = int(width / down_scale), int(height / down_scale)
        self.umap = np.array([[j for _ in range(self.down_width)] for j in range(self.down_height)])
        self.vmap = np.array([[i for i in range(self.down_width)] for _ in range(self.down_height)])
        self.intrinsics = intrinsics
        self.cloud = pcl.PointCloud()

    def detect(self, depth, observation_mask):

        # additional parameters
        initial_z_axis = None
        angle_eps = 10.0
        max_z = -1

        # adapt intrinsics according to image scaling
        cam_fx, cam_fy = self.intrinsics[0, 0]/self.down_scale, self.intrinsics[1, 1]/self.down_scale
        cam_cx, cam_cy = self.intrinsics[0, 2]/self.down_scale, self.intrinsics[1, 2]/self.down_scale

        # remove depth measurements for detected objects
        planar = np.float32(depth.copy())
        planar[observation_mask > 0] = 0
        # remove depth measurements of background ~ too far behind detected objects (need to be supported by plane)
        threshold = np.mean(depth[observation_mask > 0]) + np.std(depth[observation_mask > 0])
        planar[planar > threshold] = 0

        # === project depth to point cloud
        pt2 = resize(planar, (self.down_height, self.down_width), anti_aliasing=False) / 1000.0
        pt0 = (self.vmap - cam_cx) * pt2 / cam_fx
        pt1 = (self.umap - cam_cy) * pt2 / cam_fy
        points = np.dstack((pt0, pt1, pt2)).astype(np.float32).reshape((self.down_width * self.down_height, 3))

        # remove invalid points
        z_values = pt2.reshape(self.down_height * self.down_width) != 0
        if max_z > 0:
            points = points[np.logical_and(z_values > 0, z_values <= max_z)]
        else:
            points = points[z_values > 0]

        # === use PCL for plane segmentation
        self.cloud.from_array(points)
        seg = self.cloud.make_segmenter()#_normals(ksearch=100)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)#_NORMAL_PLANE)
        seg.set_method_type(pcl.SAC_MSAC)#RANSAC)
        seg.set_distance_threshold(0.005)#0.02)
        # seg.set_normal_distance_weight(0.5)
        # seg.set_max_iterations(1000)
        if initial_z_axis is not None:
            seg.set_axis(*initial_z_axis)
            seg.set_eps_angle(np.deg2rad(angle_eps))
        indices, coefficients = seg.segment()

        if len(coefficients) == 0:
            raise ValueError("no coefficients for plane - none detected")

        # === plane coefficients to transformation matrix: adapted from https://math.stackexchange.com/a/1957132
        # R: compute basis vectors from n
        if coefficients[2] > 0:  # make sure we're facing camera
            coefficients = [-c for c in coefficients]

        n = np.array(coefficients[:3]) / np.linalg.norm(coefficients[:3])
        nxy_norm = np.linalg.norm(n[:2])
        R = np.eye(3)
        # - b1 vector orthogonal to n
        R[0, 0] = n[1] / nxy_norm
        R[1, 0] = -n[0] / nxy_norm
        R[2, 0] = 0
        # - b2: vector normal to n and b1 -- n x b1 (-> right-handed)
        R[0, 1] = n[0] * n[2] / nxy_norm
        R[1, 1] = n[1] * n[2] / nxy_norm
        R[2, 1] = -nxy_norm
        # - b3: the normal vector itself
        R[:, 2] = n[:3]
        R = np.matrix(R)

        # t: move -d in direction of n
        t = -np.matrix(n * coefficients[3])

        # compose final matrix
        T = np.eye(4, 4)
        T[:3, :3] = R
        T[:3, 3] = t

        return np.matrix(T)
