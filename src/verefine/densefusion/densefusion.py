# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import numpy as np
import numpy.ma as ma
from scipy.spatial.transform.rotation import Rotation
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix


class DenseFusion:
    """
    Code adapted and extended from DenseFusion implementation.
    """

    def __init__(self, width, height):
        self.width, self.height = width, height

        # default from DenseFusion for YCB-Video
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # u,v coordinates (used in mapping)
        self.umap = np.array([[j for _ in range(self.width)] for j in range(self.height)])
        self.vmap = np.array([[i for i in range(self.width)] for _ in range(self.height)])

        # YCBV specific DenseFusion setting
        self.num_objects = 21
        self.num_points = 1000
        self.bs = 1

        # init networks
        self.estimator = PoseNet(num_points=self.num_points, num_obj=self.num_objects)
        self.estimator.cuda()
        self.estimator.load_state_dict(torch.load("/verefine/data/densefusion_estimate_ycbv.pth"))
        self.estimator.eval()

        self.refiner = PoseRefineNet(num_points=self.num_points, num_obj=self.num_objects)
        self.refiner.cuda()
        self.refiner.load_state_dict(torch.load("/verefine/data/densefusion_refine_ycbv.pth"))
        self.refiner.eval()

    def estimate(self, rgb, depth, intrinsics, roi, mask, obj_id, hypotheses_per_instance=1):
        # get mask (from segmentation + valid depth values)
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(mask, 1))
        mask = mask_label * mask_depth

        # get samples in mask
        vmin, umin, vmax, umax = roi
        choose = mask[vmin:vmax, umin:umax].flatten().nonzero()[0]
        if len(choose) > self.num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        elif len(choose) == 0:
            raise ZeroDivisionError()
        else:
            choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

        # point cloud from depth -- image space (u, v, D) to camera space (X, Y, D)
        D_masked = depth[vmin:vmax, umin:umax].flatten()[choose][:, np.newaxis].astype(np.float32)
        u_masked = self.umap[vmin:vmax, umin:umax].flatten()[choose][:, np.newaxis].astype(np.float32)
        v_masked = self.vmap[vmin:vmax, umin:umax].flatten()[choose][:, np.newaxis].astype(np.float32)

        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        X_masked = (v_masked - cx) * D_masked / fx
        Y_masked = (u_masked - cy) * D_masked / fy
        cloud = np.concatenate((X_masked, Y_masked, D_masked), axis=1)

        # rgb image (in mask)
        rgb_masked = np.array(rgb)[:, :, :3]
        rgb_masked = np.transpose(rgb_masked, (2, 0, 1))
        rgb_masked = rgb_masked[:, vmin:vmax, umin:umax]

        # prepare for pose estimation
        cloud = torch.from_numpy(cloud.astype(np.float32))
        choose = torch.LongTensor(np.array([choose]).astype(np.int32))
        rgb_masked = self.norm(torch.from_numpy(rgb_masked.astype(np.float32)))
        index = torch.LongTensor([obj_id - 1])

        cloud = Variable(cloud).cuda()
        choose = Variable(choose).cuda()
        rgb_masked = Variable(rgb_masked).cuda()
        index = Variable(index).cuda()

        cloud = cloud.view(1, self.num_points, 3)
        rgb_masked = rgb_masked.view(1, 3, rgb_masked.size()[1], rgb_masked.size()[2])

        # run pose estimation
        pred_r, pred_t, pred_c, emb = self.estimator(rgb_masked, cloud, choose, index)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)

        pred_c = pred_c.view(self.bs, self.num_points)
        pred_t = pred_t.view(self.bs * self.num_points, 1, 3)
        points = cloud.view(self.bs * self.num_points, 1, 3)

        # select max conf hypothesis and [hypotheses_per_object-1] uniformly sampled additional hypotheses
        how_max, which_max = pred_c.max(1)
        candidates = list(range(self.num_points))
        candidates.remove(which_max)  # so this is not selected twice
        selected = np.random.choice(candidates, hypotheses_per_instance - 1, replace=False)  # uniform w/o replace
        which_max = torch.cat((which_max, torch.LongTensor(selected.astype(np.int64)).cuda()))

        hypotheses = []
        for hypothesis_id in which_max:
            instance_r = pred_r[0][hypothesis_id].view(-1).cpu().data.numpy()  # w, x, y, z
            instance_r = np.concatenate((instance_r[1:], [instance_r[0]]))  # to x, y, z, w
            instance_t = (points + pred_t)[hypothesis_id].view(-1).cpu().data.numpy()

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = Rotation.from_quat(instance_r).as_matrix()
            pose[:3, 3] = instance_t.squeeze()

            hypothesis = {
                'obj_id': str(obj_id),
                'pose': pose,
                'confidence': pred_c[0][hypothesis_id],
                'obj_mask': mask,
                'emb': emb,
                'cloud_obs': cloud
            }
            hypotheses.append(hypothesis)

        return hypotheses

    def refine(self, obj_id, pose, emb, cloud, iterations=1):
        class_id = int(obj_id) - 1
        index = torch.LongTensor([class_id])
        index = Variable(index).cuda()

        pose_ref = pose.copy()
        for _ in range(iterations):
            # transform cloud according to pose estimate
            R = Variable(torch.from_numpy(pose_ref[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            ts = Variable(torch.from_numpy(pose_ref[:3, 3].astype(np.float32))).cuda().view(1, 3)\
                .repeat(self.num_points, 1).contiguous().view(1, self.num_points, 3)
            new_cloud = torch.bmm((cloud - ts), R).contiguous()

            # predict delta pose
            pred_q, pred_t = self.refiner(new_cloud, emb, index)
            pred_q = pred_q.view(1, 1, -1)
            pred_q = pred_q / (torch.norm(pred_q, dim=2).view(1, 1, 1))
            pred_q = pred_q.view(-1).cpu().data.numpy()
            pred_t = pred_t.view(-1).cpu().data.numpy()

            # apply delta to get new pose estimate
            pose_delta = quaternion_matrix(pred_q)
            pose_delta[:3, 3] = pred_t
            pose_ref = pose_ref @ pose_delta

        return pose_ref

