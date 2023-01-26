# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import numpy as np
import json
import trimesh
import cv2 as cv


class YcbvDataset:

    def __init__(self):
        # verefine.json - meta
        meta_vf = json.load(open("/verefine/data/ycbv_verefine.json", 'r'))
        self.baseline = "df"

        # intrinsics, depth scale, model infos (offset, com, collider) - from meta
        # camera data
        self.width, self.height = meta_vf['im_size']
        self.camera_intrinsics = np.asarray(meta_vf['intrinsics']).reshape(3, 3)
        self.depth_scale = meta_vf['depth_scale']

        # object meta
        obj_meta = meta_vf['objects']
        self.num_objects = len(obj_meta)
        self.obj_ids = list(obj_meta.keys())
        self.obj_names = dict(zip(self.obj_ids, [obj_meta[obj_id]['name'] for obj_id in self.obj_ids]))
        self.obj_coms = [obj_meta[obj_id]['offset_center_mass'] for obj_id in self.obj_ids]
        self.obj_model_offset = [obj_meta[obj_id]['offset_bop'] for obj_id in self.obj_ids]
        self.obj_scale = meta_vf['object_scale']

        # meshes for simulation and rendering
        self.collider_paths = ["/verefine/data/models/simulate/obj_{id:06}.obj".format(id = int(obj_id)) for obj_id in self.obj_ids]
        scale = np.diag([self.obj_scale]*3 + [1.0])
        self.meshes = [trimesh.load("/verefine/data/models/render/obj_{id:06}.ply".format(id = int(obj_id))).apply_transform(scale)
                       for obj_id in self.obj_ids]

    @staticmethod
    def get_normal_image(depth):
        D = depth.copy()

        # inpaint missing depth values
        D = cv.inpaint(D.astype(np.float32), np.uint8(D == 0), 3, cv.INPAINT_NS)
        # smoothen depth map
        blur_size = (9, 9)
        D = cv.GaussianBlur(D, blur_size, sigmaX=10.0)

        # get derivatives
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        dzdx = cv.filter2D(D, -1, kernelx)
        dzdy = cv.filter2D(D, -1, kernely)

        # gradient ~ normal
        normal = np.dstack((dzdy, dzdx, D != 0.0))  # only where we have a depth value
        n = np.linalg.norm(normal, axis=2)
        n = np.dstack((n, n, n))
        normal = np.divide(normal, n, where=(n != 0))

        # remove invalid values
        normal[n == 0] = 0.0
        normal[depth == 0] = 0.0
        return normal
