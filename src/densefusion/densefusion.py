import copy
import numpy as np
import numpy.ma as ma
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms

import sys
sys.path.append("/verefine/3rdparty/DenseFusion")
sys.path.append("/verefine/3rdparty/DenseFusion/lib")
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix

from src.verefine.refiner_interface import Refiner


class DenseFusion(Refiner):
    """
    Code adapted and extended from DenseFusion implementation.
    """

    def __init__(self, width, height, intrinsics, dataset, only_estimator=False, mode="base"):
        Refiner.__init__(self, intrinsics, dataset, mode=mode)

        self.width, self.height = width, height
        self.dataset = dataset

        # default from DenseFusion for YCB-Video and LINEMOD
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # u,v coordinates (used in mapping)
        self.umap = np.array([[j for _ in range(self.width)] for j in range(self.height)])
        self.vmap = np.array([[i for i in range(self.width)] for _ in range(self.height)])

        # dataset specific DenseFusion setting
        self.num_points = 1000
        self.refinement_steps = 2
        self.bs = 1

        # init networks
        estimate_model = '/densefusion/data/pose_model_26_0.012863246640872631.pth'
        self.estimator = PoseNet(num_points=self.num_points, num_obj=self.dataset.num_objects)
        self.estimator.cuda()
        self.estimator.load_state_dict(torch.load(estimate_model))
        self.estimator.eval()

        if not only_estimator:
            refine_model = "/densefusion/data/pose_refine_model_69_0.009449292959118935.pth"  # TODO
            self.refiner = PoseRefineNet(num_points=self.num_points, num_obj=self.dataset.num_objects)
            self.refiner.cuda()
            self.refiner.load_state_dict(torch.load(refine_model))
            self.refiner.eval()
        self.only_estimator = only_estimator

    def forward(self, rgb, depth, intrinsics, roi, mask, class_id):
        """
        RGBD+detection to predictions, sampled point cloud and color embedding
        :param rgb:
        :param depth:
        :param intrinsics:
        :param roi:
        :param mask:
        :param class_id:
        :return:
        """
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
        D_masked = D_masked / 1000.0  # observation in mm -- convert to meters
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
        index = torch.LongTensor([class_id - 1])

        cloud = Variable(cloud).cuda()
        choose = Variable(choose).cuda()
        rgb_masked = Variable(rgb_masked).cuda()
        index = Variable(index).cuda()

        cloud = cloud.view(1, self.num_points, 3)
        rgb_masked = rgb_masked.view(1, 3, rgb_masked.size()[1], rgb_masked.size()[2])

        # run pose estimation
        pred_r, pred_t, pred_c, emb = self.estimator(rgb_masked, cloud, choose, index)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)

        return pred_r, pred_t, pred_c, emb, cloud

    def estimate(self, rgb, depth, intrinsics, roi, mask, class_id, hypotheses_per_instance=1):
        """

        :param rgb:
        :param depth:
        :param intrinsics:
        :param roi:
        :param mask:
        :param class_id:
        :param hypotheses_per_instance:
        :return:
        """

        # --- object pose estimation
        hypotheses = []
        try:
            pred_r, pred_t, pred_c, _, cloud = self.forward(rgb, depth, intrinsics, roi, mask, class_id)

            pred_c = pred_c.view(self.bs, self.num_points)
            pred_t = pred_t.view(self.bs * self.num_points, 1, 3)
            points = cloud.view(self.bs * self.num_points, 1, 3)

            # select max conf hypothesis and [hypotheses_per_object-1] uniformly sampled additional hypotheses
            how_max, which_max = pred_c.max(1)
            candidates = list(range(self.num_points))
            candidates.remove(which_max)  # so this is not selected twice
            selected = np.random.choice(candidates, hypotheses_per_instance - 1, replace=False)  # uniform w/o replace
            which_max = torch.cat((which_max, torch.LongTensor(selected.astype(np.int64)).cuda()))

            for hypothesis_id in range(hypotheses_per_instance):
                instance_r = pred_r[0][which_max[hypothesis_id]].view(-1).cpu().data.numpy()  # w, x, y, z
                instance_r = np.concatenate((instance_r[1:], [instance_r[0]]))
                instance_t = (points + pred_t)[which_max[hypothesis_id]].view(-1).cpu().data.numpy()

                # add to hypotheses
                hypotheses.append([instance_r.copy(), instance_t.copy(), how_max])
        except ZeroDivisionError:
            print("Detector lost object with id %i." % (class_id))
        return hypotheses

    def refine(self, rgb, depth, intrinsics, roi, mask, class_id, hypothesis, iterations=1):
        """

        :param rgb:
        :param depth:
        :param intrinsics:
        :param roi:
        :param mask:
        :param class_id:
        :param hypothesis:
        :param iterations:
        :return:
        """

        if self.only_estimator:
            raise NotImplementedError("'only_estimator' mode requested")

        my_r, my_t, _ = hypothesis
        my_r = np.concatenate(([my_r[3]], my_r[:3]))  # to w, x, y, z
        _, _, _, emb, cloud = self.forward(rgb, depth, intrinsics, roi, mask, class_id)

        index = torch.LongTensor([class_id - 1])
        index = Variable(index).cuda()

        # refine [refinement_steps] times
        for ite in range(0, iterations):
            # TODO clean this up
            num_points = 1000  # TODO
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3) \
                .repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0][3] = my_t[0]
            my_mat[1][3] = my_t[1]
            my_mat[2][3] = my_t[2]
            new_cloud = torch.bmm((cloud - T), R).contiguous()

            ref_pred_r, ref_pred_t = self.refiner(new_cloud, emb, index)
            ref_pred_r = ref_pred_r.view(1, 1, -1)
            ref_pred_r = ref_pred_r / (torch.norm(ref_pred_r, dim=2).view(1, 1, 1))
            my_r_2 = ref_pred_r.view(-1).cpu().data.numpy()
            my_t_2 = ref_pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)

            my_mat_2[0][3] = my_t_2[0]
            my_mat_2[1][3] = my_t_2[1]
            my_mat_2[2][3] = my_t_2[2]

            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0][3] = 0
            my_r_final[1][3] = 0
            my_r_final[2][3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            my_r = my_r_final
            my_t = my_t_final

        my_r = np.concatenate((my_r[1:], [my_r[0]]))
        hypothesis[0], hypothesis[1] = my_r, my_t
        return hypothesis
