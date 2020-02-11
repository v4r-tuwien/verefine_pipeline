import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np
import json
import scipy.io as scio
from scipy.spatial.transform.rotation import Rotation
from scipy.spatial import cKDTree as KDTree
import os
import PIL
import time
import yaml
import gc
import random


# # make reproducible (works up to BAB -- TODO VF smh not)
# seed = 0
# np.random.seed(seed)
# random.seed(seed)

from src.util.fast_renderer import Renderer
from src.util.dataset import ExApcDataset
from src.verefine.simulator import Simulator
from src.verefine.verefine import Hypothesis, PhysIR, BudgetAllocationBandit, SceneBAB
import src.verefine.verefine as Verefine
from src.icp.icp import TrimmedIcp


"""
TODO

- simplified models for simulation

- --clean-up depth (could use output from Mitash)-- don't - our score seems to work better without
-- remove "already explained points"
-- remove table
- instead: use this as segmentation for ICP

- penalty for multiple assignment on scene level (similar to Aldoma)
"""

# settings
PATH_APC = "/home/dominik/experiments/PhysimGlobalPose/src/dataset/"
# PATH_APC = "/home/dominik/experiments/PhysimGlobalPose/src/dataset_baseline3_mcts-60s-default/"

POOL = "clusterPose"  # "allPose" for Super4PCS(?) ordered by LCP, "clusterPose" for cluster hypotheses (exactly 25), "super4pcs" for Super4PCS (best LCP), "search" for MCTS
ALL_COSTS = False
num_scenes = 42
PLOT = False

MODE = "BASE" if POOL in ["super4pcs", "search"] else "VFtree"  # "BASE", "PIR", "BAB", "VFlist", "VFtree"
EST_MODE = "PCS"
REF_MODE = "" if POOL in ["super4pcs", "search"] else "ICP"

obj_names = {  # TODO start from 0 or 1?
    "crayola_24_ct": 1,
    "expo_dry_erase_board_eraser": 2,
    "folgers_classic_roast_coffee": 3,
    "scotch_duct_tape": 4,
    "up_glucose_bottle": 5,
    "laugh_out_loud_joke_book": 6,
    "soft_white_lightbulb": 7,
    "kleenex_tissue_box": 8,
    "dove_beauty_bar": 9,
    "elmers_washable_no_run_school_glue": 10,
    "rawlings_baseball": 11
}

# TODO dependency order in gt_info != hardcoded order in Mitash
# ind_mitash = [[1, 2], [1, 1, 1], [1, 2], [2, 1], [2, 1], [1, 2], [1, 2], [1, 2], [1, 2], [2, 1], [1, 2], [1, 2],
#               [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
#               [1, 1, 1], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
ind_mitash = [[1, 2], [1, 1, 1], [1, 2], [2, 1], [2, 1], [1, 2], [1, 2], [1, 1, 1], [1, 2], [1, 2], [1, 2], [1, 2],  # --12
              [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
              [1, 1, 1], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
ind = 0
if POOL in ["super4pcs", "search"]:
    ind_1 = []
    for scene in range(1, min(30, num_scenes)+1):
        # with open(PATH_APC + "scene-%0.4d/gt_info.yml" % scene, 'r') as file:
        #     gt_info = yaml.load(file)
        # for v in gt_info['scene']['dependency_order']:
        #     v = len(v)
        for v in ind_mitash[scene - 1]:
            if v == 1:
                ind_1.append(ind)
            ind += v
    ind_2 = [ind for ind in range(min(num_scenes*3, 90)) if ind not in ind_1]  # indices start from 0
    ind_3 = list(range(90, min(num_scenes*3, 126)))
else:
    ind_1 = []
    ind_2 = []
    ind_3 = []

# if POOL == "search" and os.path.exists(PATH_APC + "scene-0001/duration.txt"):
#     times = []
#     for scene in range(1, num_scenes+1):
#         with open(PATH_APC + "scene-%0.4d/duration.txt" % scene, 'r') as file:
#             t = file.readlines()[-1]
#         parts = t.split(" ")
#         times.append([float(v.replace(" ", "").replace("\n", "")) for v in parts])
#     print("duration (search): %0.1f" % np.array(times)[:, 2].mean())


def cost_ADD(model_points, T_gt, T_est, symmetric=False):

    pred_t = T_est[:3, 3].T
    pred_r = T_est[:3, :3]
    target_t = T_gt[:3, 3].T
    target_r = T_gt[:3, :3]

    pred = np.dot(model_points, pred_r.T)
    pred = np.add(pred, pred_t)
    target = np.dot(model_points, target_r.T)
    target = np.add(target, target_t)

    if symmetric:
        target_tree = KDTree(target)
        closest_dists, indices = target_tree.query(pred, k=1)
    else:
        closest_dists = np.linalg.norm(pred - target, axis=1)
    return np.mean(closest_dists)


def cost_VSD(dist_test, dist_gt, dist_est, tau=20, delta=15):
    # --- visibility masks
    # "defined as a set of pixels where the surface of M_gt is in front of the scene surface, or at most by a tolerance
    # delta behind" - Hodan et al., On Evaluation of 6D Object Pose Estimation
    gt_fully_visible_count = np.count_nonzero(dist_gt)
    gt_mask_valid = np.logical_and(dist_test > 0, dist_gt > 0)
    gt_d_diff = dist_gt.astype(np.float32) - dist_test.astype(np.float32)
    gt_visib_mask = np.logical_and(gt_d_diff <= delta, gt_mask_valid)
    gt_occluded_count = np.count_nonzero(gt_visib_mask)
    gt_visibility = gt_occluded_count / gt_fully_visible_count if gt_fully_visible_count > 0 else 0.0

    # "In  addition  to  that,  to  ensure  that  the  visible  surface  of  the  sought object captured in does not
    # occlude the surface of M_est, all object pixels which are included in V_gt are added to V_est, regardless of
    # the surface distance at these pixels.", Hodan et al., On Evaluation of 6D Object Pose Estimation
    render_mask_valid = np.logical_and(dist_test > 0, dist_est > 0)
    render_d_diff = dist_est.astype(np.float32) - dist_test.astype(np.float32)
    render_visib_mask = np.logical_and(render_d_diff <= delta, render_mask_valid)
    render_visib_mask = np.logical_or(render_visib_mask, np.logical_and(gt_visib_mask, dist_est > 0))

    # intersection and union of visibility masks
    visib_inter = np.logical_and(gt_visib_mask, render_visib_mask)
    visib_union = np.logical_or(gt_visib_mask, render_visib_mask)

    # pixel-wise matching cost
    cost = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    cost = cost >= tau

    # visible surface discrepancy
    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()
    if visib_union_count > 0:
        vsd = (cost.sum() + visib_comp_count) / float(visib_union_count)
    else:
        vsd = 1.0

    return vsd, gt_visibility


def compute_mAP(errors, max_value, linestyle=None):
    values = errors.copy()
    values[values > max_value] = np.infty
    values = np.sort(values)
    n = values.shape[0]
    accuracy = np.cumsum(np.ones((1, n))) / n

    valid = np.logical_not(np.isinf(values))
    rec = values[valid]
    prec = accuracy[valid]

    if len(prec) == 0:
        return -1

    mrec = np.concatenate(([0], rec, [max_value]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))

    for i in range(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])

    if linestyle is not None:
        plt.plot(mrec, mpre, linestyle)

    i = np.argwhere((mrec[1:] != mrec[0:-1]) == 1) + 1
    scale = 1 / max_value
    return np.dot((mrec[i] - mrec[i - 1]).T, mpre[i]) * scale


# -----------------

if __name__ == "__main__":

    Verefine.HYPOTHESES_PER_OBJECT = 25
    Verefine.ITERATIONS = 1
    Verefine.SIM_STEPS = 60
    Verefine.C = np.sqrt(2)

    dataset = ExApcDataset(base_path=PATH_APC)
    ref = None
    durations = []
    errors_translation, errors_rotation = [], []
    errors_ssd, errors_adi, errors_vsd = [], [], []

        # if MODE != "BASE" or REF_MODE == "ICP":
    renderer = Renderer(dataset, recompute_normals=True)
    Verefine.RENDERER = renderer
        # renderer.create_egl_context()

    if MODE != "BASE":
        simulator = Simulator(dataset, instances_per_object=Verefine.HYPOTHESES_PER_OBJECT)
        Verefine.SIMULATOR = simulator
        pir = None

    with open(PATH_APC + "/obj_config_apc.yml", 'r') as file:
        obj_config = yaml.load(file)
    symInfos = {}
    for k, v in obj_config['objects'].items():
        if not "object_" in k:
            continue
        symInfos[v['classId']] = v['symmetry']

    # # convert to BOP format
    # model_info = {}
    # for i in range(0, 11):
    #     mins = list((dataset.pcd[i].min(axis=0) * 1000).flat)
    #     sizes = list(dataset.pcd[i].max(axis=0) * 1000 - dataset.pcd[i].min(axis=0) * 1000)
    #
    #     from numpy import random, nanmax, argmax, unravel_index
    #     from scipy.spatial.distance import pdist, squareform
    #
    #     A = dataset.pcd[i]
    #     D = pdist(A)
    #     D = squareform(D)
    #     diameter, [I_row, I_col] = nanmax(D), unravel_index(argmax(D), D.shape)
    #     diameter *= 1000
    #
    #
    #     def sym_steps(nfo):
    #         if nfo == 90:
    #             n = 4
    #             d = np.deg2rad(90)
    #         elif nfo == 180:
    #             n = 2
    #             d = np.deg2rad(180)
    #         elif nfo == 360:
    #             n = 1
    #             d = np.deg2rad(360)
    #         else:
    #             n = 1
    #             d = 0
    #         return n, d
    #
    #
    #     xn, dx = sym_steps(symInfos[i + 1][0])
    #     yn, dy = sym_steps(symInfos[i + 1][1])
    #     zn, dz = sym_steps(symInfos[i + 1][2])
    #     symmetries_continuous = []
    #     symmetries_discrete = []
    #     for x in range(xn):
    #         if x == 0 and dx == np.deg2rad(360):  # continuous
    #             symmetries_continuous.append(
    #                 {
    #                     "axis": [1, 0, 0],
    #                     "offset": [0, 0, 0]
    #                 }
    #             )
    #
    #         for y in range(yn):
    #             if y == 0 and dy == np.deg2rad(360):  # continuous
    #                 symmetries_continuous.append(
    #                     {
    #                         "axis": [0, 1, 0],
    #                         "offset": [0, 0, 0]
    #                     }
    #                 )
    #
    #             for z in range(zn):
    #                 if z == 0 and dz == np.deg2rad(360):  # continuous
    #                     symmetries_continuous.append(
    #                         {
    #                             "axis": [0, 0, 1],
    #                             "offset": [0, 0, 0]
    #                         }
    #                     )
    #
    #                 T = np.eye(4)
    #                 T[:3, :3] = Rotation.from_euler('xyz', [x * dx, y * dy, z * dz]).as_dcm()
    #                 if not np.allclose(T, np.eye(4)):
    #                     symmetries_discrete.append(list(T.reshape(16)))
    #
    #     # print(i+1)
    #     # print(diameter)
    #     # print(mins)
    #     # print(sizes)
    #     # print("---")
    #     model_info[i + 1] = {
    #         "diameter": diameter,
    #         "min_x": mins[0],
    #         "min_y": mins[1],
    #         "min_z": mins[2],
    #         "size_x": sizes[0],
    #         "size_y": sizes[1],
    #         "size_z": sizes[2],
    #     }
    #     if len(symmetries_discrete) > 0:
    #         model_info[i + 1]["symmetries_discrete"] = symmetries_discrete
    #     if len(symmetries_continuous) > 0:
    #         model_info[i + 1]["symmetries_continuous"] = symmetries_continuous
    # with open("/mnt/Data/datasets/BOP19/exapc/models/models_info.json", 'w') as file:
    #     json.dump(model_info, file)
    #
    # import sys
    # sys.exit(0)

    # loop over scenes...
    objects = []
    scenes = list(range(1, num_scenes+1))  # all in Mitash' script -> same as in paper
    # scenes = [2, 25]  # no dependencies
    # scenes = [1] + list(range(3, 25)) + list(range(26, 31))  # 2-obj
    # scenes = list(range(31, 36))  # 3-obj (until 35...)
    # scenes = list(range(31, 43))  # 3-obj (all in dataset)
    # scenes = list(range(1, 36))  # all with annotated dependencies
    # scenes = list(range(1, 43))  # all scenes in dataset
    obj_ids = []
    obj_gt_Ts = {}
    scene_camera = {}

    for scene in scenes:
        # if scene < 31:  #> 30:#
        #     errors_translation += [0, 0, 0]
        #     errors_rotation += [0, 0, 0]
        #     errors_ssd += [0, 0, 0]
        #     errors_adi += [0, 0, 0]
        #     errors_vsd += [0, 0, 0]
        #     ind += 3
        #     continue
        print("scene %i..." % scene)

        # gt info
        with open(PATH_APC + "scene-%0.4d/gt_info.yml" % scene, 'r') as file:
            gt_info = yaml.load(file)

        # load observation
        rgb = np.array(PIL.Image.open(PATH_APC + "scene-%0.4d/frame-000000.color.png" % scene))
        depth = np.array(PIL.Image.open(PATH_APC + "scene-%0.4d/frame-000000.depth.png" % scene), dtype=np.uint16)
        depth = (depth << 13 | depth >> 3)/10  # [mm]
        full_depth = depth.copy()
        scene_depth = np.array(PIL.Image.open(PATH_APC + "scene-%0.4d/debug_search/scene.png" % scene), dtype=np.uint16)  # table removed (as in Mitash)
        scene_depth = (scene_depth << 13 | scene_depth >> 3) / 10  # [mm]

        # # convert to BOP format
        # if not os.path.exists("/mnt/Data/datasets/BOP19/exapc/test/%0.6d" % scene):
        #     os.mkdir("/mnt/Data/datasets/BOP19/exapc/test/%0.6d" % scene)
        #     os.mkdir("/mnt/Data/datasets/BOP19/exapc/test/%0.6d/rgb" % scene)
        #     os.mkdir("/mnt/Data/datasets/BOP19/exapc/test/%0.6d/depth" % scene)
        #     os.mkdir("/mnt/Data/datasets/BOP19/exapc/test/%0.6d/mask" % scene)
        #     os.mkdir("/mnt/Data/datasets/BOP19/exapc/test/%0.6d/mask_visib" % scene)
        # PIL.Image.fromarray(rgb.astype(np.uint8)).save(
        #     "/mnt/Data/datasets/BOP19/exapc/test/000001/rgb/%0.6d.png" % scene)
        # PIL.Image.fromarray(depth.astype(np.uint16)).save(
        #     "/mnt/Data/datasets/BOP19/exapc/test/000001/depth/%0.6d.png" % scene)

        # camera data
        camera_intrinsics = np.array(gt_info['camera']['camera_intrinsics'])
        camera_extrinsics = np.matrix(np.eye(4))
        # note: translation relative to table (rotation of table is always I)
        # camera_extrinsics[:3, 3] = np.matrix(gt_info['camera']['camera_pose'][:3]).T
        camera_extrinsics[:3, 3] = (np.matrix(gt_info['camera']['camera_pose'][:3])).T
        if POOL == "clusterPose":
            camera_extrinsics[:3, 3] -= np.matrix(gt_info['rest_surface']['surface_pose'][:3]).T  # [m]
        camera_q = gt_info['camera']['camera_pose'][3:]  # wxyz
        camera_q = camera_q[1:] + [camera_q[0]]  # xyzw
        camera_extrinsics[:3, :3] = Rotation.from_quat(camera_q).as_dcm()

        # # convert to BOP format
        # scene_camera["%i" % scene] = {
        #     "cam_K": list(camera_intrinsics.reshape(9)),
        #     "cam_R_w2c": list(np.array(camera_extrinsics[:3, :3]).reshape(9)),
        #     "cam_t_w2c": list(np.array(camera_extrinsics[:3, 3]).reshape(3)*1000),
        #     "depth_scale": 1.0
        # }


        def estimate_normals(D):
            D_px = D.copy() * camera_intrinsics[0, 0]  # from meters to pixels

            import cv2 as cv

            # inpaint missing depth values
            D_px = cv.inpaint(D_px.astype(np.float32), np.uint8(D_px == 0), 3, cv.INPAINT_NS)
            # blur
            D_px = cv.GaussianBlur(D_px, (9, 9), sigmaX=10.0)

            # get derivatives

            # # 1) Sobel
            # dzdx = cv.Sobel(depth, cv.CV_64F, dx=1, dy=0, ksize=-1)  # difference
            # dzdy = cv.Sobel(depth, cv.CV_64F, dx=0, dy=1, ksize=-1)  # step size of 1px

            # 2) Prewitt
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
            normal[D == 0] = 0.0

            # plt.imshow((normal + 1) / 2)
            # plt.show()
            return normal

        scene_normals = estimate_normals(scene_depth / 1000)
        # plt.imshow((1+scene_normals)/2)
        # plt.show()

        # # to old format
        # with open(PATH_APC_old + "scene-%0.4d/cameraExtrinsic.txt" % scene, 'w') as file:
        #     file.write(str(camera_extrinsics).replace("[", "").replace("]", ""))
        # with open(PATH_APC_old + "scene-%0.4d/cameraIntinsic.txt" % scene, 'w') as file:
        #     file.write(str(camera_intrinsics).replace("[", "").replace("]", ""))

        if ref is None:
            # if REF_MODE == "ICP":
            ref = TrimmedIcp(renderer, camera_intrinsics, dataset, mode="bab")
            # else:
            #     ref = None  #raise NotImplementedError("only TrICP available atm")

            if MODE != "BASE":
                pir = PhysIR(ref, simulator)
                Verefine.REFINER = pir

        # get pose estimates
        # obj_ids = []
        # obj_gt_Ts = {}
        hypotheses = []
        obj_depths = []
        n_obj = 0

        # TODO still a bit leaky -- check using fast.ai GPUMemTrace
        gc.collect()

        # get gt objects
        for i_obj in range(gt_info['scene']['num_objects']):
            # -- infos
            obj_info = gt_info['scene']['object_%i' % (i_obj+1)]

            obj_name = obj_info['name']
            obj_id = obj_names[obj_name]

            # # to old format
            # with open(PATH_APC_old + "scene-%0.4d/objects.txt" % scene, 'a') as file:
            #     file.write(obj_name + "\n")
            #     # file.write("")
            # continue
            # # break

            obj_gt_T = np.matrix(np.eye(4))  # in world coordinates
            # obj_gt_T[:3, 3] = np.matrix(obj_info['pose'][:3]).T
            obj_gt_T[:3, 3] = (np.matrix(obj_info['pose'][:3])).T
            obj_q = obj_info['pose'][3:]  # wxyz
            obj_q = obj_q[1:] + [obj_q[0]]  # xyzw
            obj_gt_T[:3, :3] = Rotation.from_quat(obj_q).as_dcm()

            # # to old format
            # with open(PATH_APC_old + "scene-%0.4d/gt_pose_%s.txt" % (scene, obj_name), 'w') as file:
            #     file.write(str(obj_gt_T[:3, :]).replace("[", "").replace("]", "").replace("\n", "") + "\n")
            # continue

            # obj_gt_T[:3, 3] = np.matrix(obj_info['pose'][:3]).T - obj_gt_T[:3, :3]*np.matrix(gt_info['rest_surface']['surface_pose'][:3]).T
            world_to_cam = camera_extrinsics.copy()
            if POOL == "clusterPose":
                world_to_cam[:3, 3] += np.matrix(gt_info['rest_surface']['surface_pose'][:3]).T  # [m]
            world_to_cam[:3, :3] = world_to_cam[:3, :3].T
            world_to_cam[:3, 3] = -world_to_cam[:3, :3] * world_to_cam[:3, 3]
            # world_to_cam = camera_extrinsics.I
            obj_gt_T = world_to_cam * obj_gt_T  # to camera coordinates TODO correct?

            obj_ids.append(obj_id)
            # obj_gt_Ts["%i-%i" % (scene, obj_id)] = obj_gt_T  # BOP format
            obj_gt_Ts[obj_id] = obj_gt_T
            # continue

            # -- hyp
            if POOL != "search":
                path = PATH_APC + "scene-%0.4d/debug_super4PCS/%s_%s.txt" % (scene, POOL, obj_name)
            else:
                path = PATH_APC + "scene-%0.4d/debug_search/after_search_%s.txt" % (scene, obj_name)
            if not os.path.exists(path):
                # TODO count as FN
                continue  # no hypotheses for this object
            else:
                n_obj += 1
            with open(path, 'r') as file:
                obj_hypotheses = file.readlines()

            if POOL == "clusterPose":#"#not in ["super4pcs", "search"]:
                # add cluster hypotheses
                with open(PATH_APC + "scene-%0.4d/debug_super4PCS/%s_%s.txt"
                          % (scene, POOL.replace("Pose", "Score"), obj_name), 'r') as file:
                    obj_hypotheses_scores = file.readlines()
                    obj_hypotheses_scores = [float(v.replace("\n", "")) for v in obj_hypotheses_scores]
                score_order = np.argsort(obj_hypotheses_scores)[::-1]
                obj_hypotheses_scores = [obj_hypotheses_scores[i] for i in score_order]
                obj_hypotheses = [obj_hypotheses[i] for i in score_order]

                # # add best super4pcs hypothesis (as in Mitash)
                # path_super4pcs = PATH_APC + "scene-%0.4d/debug_super4PCS/super4pcs_%s.txt" % (scene, obj_name)
                # with open(path, 'r') as file:
                #     obj_hypotheses_super4pcs = file.readlines()
                # obj_hypotheses = [obj_hypotheses_super4pcs[-1]] + obj_hypotheses
                # obj_hypotheses_scores = [1.0] + obj_hypotheses_scores
            elif POOL:
                obj_hypotheses = [obj_hypotheses[-1]]  # super4pcs: 0 base, 1 w/ ICP; search: -1 is best
                obj_hypotheses_scores = [1.0]

            # mask + roi
            obj_mask = np.array(PIL.Image.open(PATH_APC + "scene-%0.4d/debug_super4PCS/frame-000000.%s.mask.png" % (scene, obj_name)))
            # obj_mask = obj_mask / 10000  # [0, 1]
            # TODO compute mask like Mitash (threshold or argmax) + they also do some outlier removal in compute3dSegment
            # obj_mask = obj_mask > 0.2  # at least ... probability TODO which value?

            vmin, umin = np.min(np.argwhere(obj_mask > 0), axis=0)
            vmax, umax = np.max(np.argwhere(obj_mask > 0), axis=0)
            obj_roi = [vmin, umin, vmax, umax]

            obj_depth = scene_depth.copy()
            obj_depth[obj_mask == 0] = 0
            obj_depths.append(obj_depth)

            # # scene_depth_down = None
            # # downsample and mask observed depth
            # depth_pcd = ref.depth_to_cloud(obj_depth, camera_intrinsics, obj_mask)
            # import open3d as o3d
            # pcd, pcd_down = o3d.PointCloud(), o3d.PointCloud()
            # pcd.points = o3d.Vector3dVector(depth_pcd / 1000)
            # pcd_down = o3d.voxel_down_sample(pcd, voxel_size=0.005)
            # scene_depth_down = np.array(pcd_down.points)
            # print(scene_depth_down.shape)

            # or: load downsampled segment
            with open(
                PATH_APC + "scene-%0.4d/debug_super4PCS/pclSegment_%s.ply"
                % (scene, obj_name)
                , 'r'
            ) as file:
                pcd = file.readlines()
            pcd_seg = []
            started = False
            for line in pcd[:-1]:  # last line is meta
                if not started:
                    if not line.startswith("end_header"):
                        continue
                    else:
                        started = True
                        continue
                parts = line.split(" ")
                parts = [float(v.replace("\n", "")) for v in parts]
                pcd_seg.append(parts)
            scene_depth_down = np.array(pcd_seg)
            # print(scene_depth_down.shape)
            #
            # import open3d as o3d
            # pcd = o3d.PointCloud()
            # pcd.points = o3d.Vector3dVector(scene_depth_down)
            # pcd, ind = o3d.statistical_outlier_removal(pcd, 16, 1.0)
            # # pcd, ind = o3d.radius_outlier_removal(pcd, 4, 0.01)
            # scene_depth_down = np.array(pcd.points)
            # # print(scene_depth_down.shape)


            # hypotheses
            new_hypotheses = []
            confidences = []
            for hi, (obj_hypothesis, c) in enumerate(zip(obj_hypotheses[:Verefine.HYPOTHESES_PER_OBJECT],
                                                         obj_hypotheses_scores[:Verefine.HYPOTHESES_PER_OBJECT])):
                parts = obj_hypothesis.split(" ")
                parts = [float(v.replace("\n", "")) for v in parts]
                # q = np.array(parts[4:-1] + [parts[3]])
                # t = np.array(parts[:3])
                # c = parts[-1]  # high is good

                obj_T = np.matrix(np.eye(4))
                # obj_T[:3, :3] = Rotation.from_quat(q).as_dcm()
                # obj_T[:3, 3] = t.reshape(3, 1)
                obj_T[:3, :] = np.array(parts).reshape(3, 4)  # in world coords
                obj_T = world_to_cam * obj_T  # to cam coords
                # estimate = [q, t, c]

                # obj_T = obj_gt_T  # TODO debug GT

                estimate = [Rotation.from_dcm(obj_T[:3, :3]).as_quat(), obj_T[:3, 3], c]

                # TODO depth, scene_depth or obj_depth?
                refiner_param = [rgb, depth, camera_intrinsics, None, obj_mask, obj_id, estimate,
                                 Verefine.ITERATIONS, scene_depth_down, None, None]
                new_hypotheses.append(
                    Hypothesis("%0.2d" % obj_id, obj_T, obj_roi, obj_mask, None, None, hi, c,
                               refiner_param=refiner_param))
                confidences.append(c)

            # only select the [HYPOTHESES_PER_OBJECT] best hypotheses
            best_estimates = np.argsort(confidences)[::-1]#[:Verefine.HYPOTHESES_PER_OBJECT]  # TODO same number as above!
            hypotheses += [[new_hypotheses[idx] for idx in best_estimates]]

            # # --- vis
            # obs = depth.reshape(480, 640, 1)
            # renderer.set_observation(obs)
            # for hypothesis in hypotheses[-1]:
            #     vis = rgb.copy()
            #     rgb_ren = []
            #
            #     rendered = hypothesis.render(observation, 'color')
            #     rgb_ren.append(rendered[0])
            #     vis[rendered[0] != 0] = vis[rendered[0] != 0] * 0.3 + rendered[0][rendered[0] != 0] * 0.7
            #
            #     def debug_draw():
            #         plt.imshow(vis)
            #
            #     drawnow(debug_draw)
            #     plt.pause(1.0)
        # # to old format
        # continue

        # --- refine
        final_hypotheses = []
        st = time.time()

        if MODE != "BASE":
            # init frame
            simulator.initialize_frame(camera_extrinsics.I)
        # if MODE != "BASE" or REF_MODE == "ICP":
        renderer.set_observation(scene_depth.reshape(480, 640, 1), scene_normals.reshape(480, 640, 3))

        if MODE == "BASE":
            # DF (base)
            refinements = 0
            # refine only max conf
            for oi, obj_hypotheses in enumerate(hypotheses):
                hypothesis = obj_hypotheses[0]  # only take best

                # refinements += refiner_param[-3]
                if REF_MODE == "ICP":
                    q, t, c = ref.refine(*hypothesis.refiner_param)

                    hypothesis.transformation[:3, :3] = Rotation.from_quat(q).as_dcm()
                    hypothesis.transformation[:3, 3] = t.reshape(3, 1)
                    hypothesis.confidence = c
                final_hypotheses.append(hypothesis)


            #     # TODO dbg -- fix GT
            #     obj_fixed_T = hypothesis.transformation
            #     obj_fixed_T = world_to_cam.I * obj_fixed_T
            #     obj_fixed_q = Rotation.from_dcm(obj_fixed_T[:3, :3]).as_quat()  # xyzw
            #     obj_fixed_q = [obj_fixed_q[3]] + list(obj_fixed_q[:-1].flat)  # wxyz
            #     obj_fixed_t = list(obj_fixed_T[:3, 3].T.flat)
            #
            #     gt_info['scene']['object_%i' % (oi + 1)]['pose_fixed'] = [float(v) for v in obj_fixed_t + obj_fixed_q]
            #     print(obj_fixed_t + obj_fixed_q)
            # with open(PATH_APC + "scene-%0.4d/gt_info_fixed.yml" % scene, 'w') as file:
            #     yaml.dump(gt_info, file)

            # print(refinements)

        elif MODE == "PIR":
            # PIR
            for obj_hypotheses in hypotheses:
                hypothesis = obj_hypotheses[0]  # only take best
                phys_hypotheses = pir.refine(hypothesis)

                final_hypotheses.append(phys_hypotheses[-1])  # pick hypothesis after last refinement step
        elif MODE == "BAB":
            refinements = 0

            with open(PATH_APC + "scene-%0.4d/gt_info.yml" % scene, 'r') as file:
                gt_info = yaml.load(file)
            trees = gt_info['scene']['dependency_order']

            for ti, tree in enumerate(trees):

                # 1) individual isolated object -> run BABn
                if len(tree) == 1:
                    ind_1.append(ind)
                    ind += 1
                    # final_hypotheses.append(None)
                    # continue
                elif len(tree) == 2:
                    ind_2 += [ind, ind + 1]
                    ind += 2
                    # final_hypotheses += [None, None]
                    # continue
                else:
                    ind_3 += [ind, ind + 1, ind + 2]
                    ind += 3
                    # final_hypotheses += [None, None, None]
                    # continue

                for i, oi in enumerate(tree):
                    obj_hypotheses = hypotheses[oi - 1]

                    # others = [j - 1 for j in tree if j != oi]
                    # mask_others = np.dstack(tuple(obj_depths))[:, :, others].sum(axis=2) > 0
                    # obj_depth = obj_depths[oi - 1].copy()
                    # unique_mask = np.logical_and(obj_depth > 0, mask_others)
                    # if (unique_mask > 0).sum() > (obj_depth > 0).sum() * 0.9:
                    #     unique_mask = obj_depth == 0
                    #
                    # d = scene_depth.copy()
                    # d[unique_mask>0] = 0

                    observation = {
                        "depth": scene_depth,
                        "normals": scene_normals,
                        "extrinsics": camera_extrinsics,
                        "intrinsics": camera_intrinsics
                    }
                    Verefine.OBSERVATION = observation
                    renderer.set_observation(scene_depth.reshape(480, 640, 1), scene_normals.reshape(480, 640, 3))

                    Verefine.fit_fn = Verefine.fit_single
                    bab = BudgetAllocationBandit(pir, observation, obj_hypotheses, unexplained=None)
                    bab.refine_max(fixed=[], unexplained=None)
                    hypothesis, plays, fit = bab.get_best()
                    assert hypothesis is not None
                    final_hypotheses.append(hypothesis)

                    refinements += bab.max_iter - len(obj_hypotheses)  # don't count initial render
        elif MODE in ["VFlist"]:
            # BAB (with PIR)
            refinements = 0

            with open(PATH_APC + "scene-%0.4d/gt_info.yml" % scene, 'r') as file:
                gt_info = yaml.load(file)
            trees = gt_info['scene']['dependency_order']

            for tree in trees:

                is_dependent = len(tree) > 1  # and MODE != "BAB"

                if is_dependent:
                    if len(tree) == 2:
                        ind_2 += [ind, ind + 1]
                        ind += 2
                        # final_hypotheses += [None, None]
                        # continue
                    else:
                        ind_3 += [ind, ind + 1, ind + 2]
                        ind += 3
                        # final_hypotheses += [None, None, None]
                        # continue
                else:
                    ind_1.append(ind)
                    ind += 1
                    # final_hypotheses.append(None)
                    # continue

                unexplained = np.ones((480, 640), dtype=np.uint8) if is_dependent else None
                fixed = []

                for i, hi in enumerate(tree):
                    obj_hypotheses = hypotheses[hi-1]

                    d = scene_depth.copy()
                    # if hi > 0:
                    # others = [j-1 for j in tree[:hi]]
                    others = [j - 1 for j in tree if j != hi]
                    mask_others = np.dstack(tuple(obj_depths))[:, :, others].sum(axis=2) > 0
                    obj_depth = obj_depths[hi - 1].copy()
                    unique_mask = np.logical_and(obj_depth > 0, mask_others)
                    if (unique_mask > 0).sum() > (obj_depth > 0).sum() * 0.9:
                        unique_mask = obj_depth == 0
                    # d[unique_mask] = 0
                    # plt.imshow(np.logical_and(obj_depth>0, np.logical_not(unique_mask>0)))
                    # plt.show()

                    # others_depth = np.ones_like(obj_depth)*1000
                    # for fixed_depth in fixed_depths:
                    #     others_depth[np.logical_and(others_depth>fixed_depth, fixed_depth > 0)] = fixed_depth[np.logical_and(others_depth>fixed_depth, fixed_depth > 0)]
                    # others_depth[others_depth == 1000] = 0
                    # plt.imshow(others_depth)
                    # plt.show()

                    observation = {
                        # "rgb": rgb,  # TODO debug
                        "depth": d,
                        "normals": scene_normals,
                        "extrinsics": camera_extrinsics,
                        "intrinsics": camera_intrinsics,
                        "mask_others": unique_mask,
                        # "depth_others": others_depth
                    }
                    Verefine.OBSERVATION = observation
                    renderer.set_observation(scene_depth.reshape(480, 640, 1),
                                             scene_normals.reshape(480, 640, 3),
                                             unique_mask.reshape(480, 640, 1))

                    # TODO check if len(tree) > 1 and i != len(tree)-1 -> only if dependencies are physical, not for occlusion!
                    Verefine.fit_fn = Verefine.fit_multi if len(tree) > 1 else Verefine.fit_single  # TODO or by mask overlap? is_dependent does not work well...
                    pir.fixed = fixed  # note: s.t. sim in initial BAB scoring is correct
                    bab = BudgetAllocationBandit(pir, observation, obj_hypotheses, unexplained=unexplained)
                    bab.refine_max(fixed=fixed, unexplained=unexplained)
                    hypothesis, plays, fit = bab.get_best()
                    assert hypothesis is not None
                    if is_dependent:
                        h_depth = hypothesis.render(observation, 'depth')[1]*1000
                        # unexplained[np.logical_and(np.abs(h_depth - scene_depth) < 8, h_depth > 0)] = 0
                        # h_depth[h_depth - scene_depth > 5] = 0
                        h_depth[np.abs(h_depth - scene_depth) > 8] = 0
                        unexplained[h_depth>0] = 0
                        obj_depths[hi-1] = h_depth.copy()
                        fixed.append([hypothesis])
                #     # fixed_depths.append(h_depth)
                    final_hypotheses.append(hypothesis)

                    refinements += bab.max_iter - len(obj_hypotheses)  # don't count initial render
        elif MODE == "VFtree":
            refinements = 0

            with open(PATH_APC + "scene-%0.4d/gt_info.yml" % scene, 'r') as file:
                gt_info = yaml.load(file)
            trees = gt_info['scene']['dependency_order']

            for ti, tree in enumerate(trees):

                # 1) individual isolated object -> run BABn
                if len(tree) == 1:
                    ind_1.append(ind)
                    ind += 1
                    # final_hypotheses.append(None)
                    # continue

                    hi = tree[0]
                    obj_hypotheses = hypotheses[hi - 1]

                    # # TODO with others still masked?
                    # others = [j - 1 for j in tree if j != hi]
                    # mask_others = np.dstack(tuple(obj_depths))[:, :, others].sum(axis=2) > 0
                    # obj_depth = obj_depths[hi - 1].copy()
                    # unique_mask = np.logical_and(obj_depth > 0, mask_others)
                    # if (unique_mask > 0).sum() > (obj_depth > 0).sum() * 0.9:
                    #     unique_mask = obj_depth == 0
                    #
                    observation = {
                        # "rgb": rgb,  # TODO debug
                        "depth": scene_depth,
                        "normals": scene_normals,
                        "extrinsics": camera_extrinsics,
                        "intrinsics": camera_intrinsics,
                        # "mask_others": unique_mask
                    }
                    Verefine.OBSERVATION = observation
                    renderer.set_observation(scene_depth.reshape(480, 640, 1), scene_normals.reshape(480, 640, 3))

                    Verefine.fit_fn = Verefine.fit_single
                    bab = BudgetAllocationBandit(pir, observation, obj_hypotheses, unexplained=None)
                    bab.refine_max(fixed=[], unexplained=None)
                    hypothesis, plays, fit = bab.get_best()
                    assert hypothesis is not None
                    final_hypotheses.append(hypothesis)

                    refinements += bab.max_iter - len(obj_hypotheses)  # don't count initial render
                # 2) dependency graph -> run tree of BABn with scene reward ~ monitored MCTS
                else:
                    if len(tree) == 2:
                        ind_2 += [ind, ind + 1]
                        ind += 2
                        # final_hypotheses += [None, None]
                        # continue
                    else:
                        ind_3 += [ind, ind + 1, ind + 2]
                        ind += 3
                        # final_hypotheses += [None, None, None]
                        # continue

                    # MCTS loop
                    # obj_masks = [obj_depths[oi-1] > 0 for oi in tree]
                    babs = [None] * len(tree)
                    scene_level_iter = 50 if len(tree) == 3 else 25
                    for si in range(scene_level_iter):
                        fixed = []

                        # build a full scene by running the BABs in order
                        selected = []
                        for i, oi in enumerate(tree):
                            obj_hypotheses = hypotheses[oi - 1]

                            # others = [j-1 for j in tree[:hi]]
                            others = [j - 1 for j in tree if j != oi]
                            mask_others = np.dstack(tuple(obj_depths))[:, :, others].sum(axis=2) > 0
                            obj_depth = obj_depths[oi - 1].copy()
                            unique_mask = np.logical_and(obj_depth > 0, mask_others)
                            if (unique_mask > 0).sum() > (obj_depth > 0).sum() * 0.9:
                                unique_mask = obj_depth == 0
                            #
                            # # plt.imshow(np.logical_and(unique_mask == 0, obj_depth > 0))  # TODO debug


                            observation = {
                                # "rgb": rgb,  # TODO debug
                                "depth": scene_depth,
                                "normals": scene_normals,
                                "extrinsics": camera_extrinsics,
                                "intrinsics": camera_intrinsics,
                                "mask_others": unique_mask
                            }
                            Verefine.OBSERVATION = observation
                            renderer.set_observation(scene_depth.reshape(480, 640, 1),
                                                     scene_normals.reshape(480, 640, 3),
                                                     unique_mask.reshape(480, 640, 1))

                            Verefine.fit_fn = Verefine.fit_multi

                            if babs[i] is None:
                                pir.fixed = fixed  # note: s.t. sim in initial BAB scoring is correct
                                babs[i] = SceneBAB(pir, observation, obj_hypotheses)

                            hi, h_sel, fit = babs[i].refine(fixed)

                            # update mask per object
                            h_depth = h_sel.render(observation, 'depth')[1] * 1000
                            h_depth[np.abs(h_depth - scene_depth) > 8] = 0  # TODO use per-pixel-fitness for this selection?
                            # obj_masks[oi - 1] = h_depth > 0  # TODO only if fit/reward is best?
                            obj_depths[oi - 1] = h_depth.copy()

                            fixed.append([h_sel])

                            selected.append((hi, h_sel, fit))

                        # compute reward
                        sel_ids = [renderer.dataset.objlist.index(int(h.id[:2])) for _, h, _ in
                                   selected if h is not None]  # TODO do this conversion in renderer
                        sel_trafos = [h.transformation for _, h, _ in selected if h is not None]

                        # a) render depth, compute score on CPU
                        observation = {
                            # "rgb": rgb,  # TODO debug
                            "depth": scene_depth,
                            "normals": scene_normals,
                            "extrinsics": camera_extrinsics,
                            "intrinsics": camera_intrinsics
                        }
                        renderer.set_observation(scene_depth.reshape(480, 640, 1),
                                                 scene_normals.reshape(480, 640, 3))
                        # rendered = renderer.render(sel_ids, sel_trafos, camera_extrinsics, camera_intrinsics, mode='depth+normal')
                        # reward = Verefine.fit_single(observation, rendered, None)
                        # reward = Verefine.fit_scene(observation, rendered)  # TODO or some global fit fn?

                        Verefine.fit_fn = Verefine.fit_single
                        _, reward = renderer.render(sel_ids, sel_trafos, camera_extrinsics, camera_intrinsics,
                                                    mode='cost')
                        # print(np.abs(reward-gpu_reward))

                        # backprop
                        for i, (hi, _, _) in enumerate(selected):
                            babs[i].backpropagate(hi, reward)

                    # get best for final estimate
                    for i in range(len(tree)):
                        # a) best object level fit
                        # hypothesis, _, _ = babs[i].get_best()  # TODO adapt to be based on reward
                        # b) best average scene level fit
                        hypothesis = babs[i].pool[np.argmax([babs[i].rewards])][-1]
                        final_hypotheses.append(hypothesis)

                    refinements += scene_level_iter * len(tree)
        #
        # elif MODE == "VF":
        #     hypotheses_pool = dict()
        #     for obj_hypotheses in hypotheses:
        #         hypotheses_pool[obj_hypotheses[0].model] = obj_hypotheses
        #
        #     Verefine.MAX_REFINEMENTS_PER_HYPOTHESIS = Verefine.ITERATIONS * Verefine.REFINEMENTS_PER_ITERATION * len(
        #         obj_hypotheses)
        #     Verefine.OBSERVATION = observation
        #     final_hypotheses, final_fit, convergence_hypotheses = Verefine.verefine_solution(hypotheses_pool)
        #
        #     # if Verefine.TRACK_CONVERGENCE:
        #         # # fill-up to 200 with final hypothesis
        #         # convergence_hypotheses += [final_hypotheses] * (200 - len(convergence_hypotheses))
        #
        #         # # write results
        #         # for convergence_iteration, iteration_hypotheses in convergence_hypotheses.items():
        #         #     iteration_hypotheses = [hs[0] for hs in iteration_hypotheses]
        #         #     for hypothesis in iteration_hypotheses:
        #         #         with open("/home/dominik/projects/hsr-grasping/convergence-vf5-c1/%0.3d_ycbv-test.csv"
        #         #                   % convergence_iteration, 'a') as file:
        #         #             parts = ["%0.2d" % scene, "%i" % 1, "%i" % int(hypothesis.model),
        #         #                      "%0.3f" % hypothesis.confidence,
        #         #                      " ".join(
        #         #                          ["%0.6f" % v for v in np.array(hypothesis.transformation[:3, :3]).reshape(9)]),
        #         #                      " ".join(
        #         #                          ["%0.6f" % (v * 1000) for v in np.array(hypothesis.transformation[:3, 3])]),
        #         #                      "%0.3f" % 1.0]
        #         #             file.write(",".join(parts) + "\n")
        #
        #     final_hypotheses = [hs[0] for hs in final_hypotheses]

        durations.append(time.time() - st)

        if not PLOT:
            for hypothesis in final_hypotheses:
                with open("/home/dominik/projects/hsr-grasping/log/%s/%s/super4pcs_exapc-test.csv"
                          % (EST_MODE, MODE if POOL == "clusterPose" else POOL), 'a') as file:
                    parts = ["1", "%i" % scene, "%i" % int(hypothesis.model), "%0.3f" % hypothesis.confidence,
                             " ".join(["%0.6f" % v for v in np.array(hypothesis.transformation[:3, :3]).reshape(9)]),
                             " ".join(["%0.6f" % (v * 1000) for v in np.array(hypothesis.transformation[:3, 3])]),
                             "%0.3f" % durations[-1]]
                    file.write(",".join(parts) + "\n")

        # --- errors
        errs_t, errs_r = [], []
        errs_ssd, errs_adi, errs_vsd = [], [], []
        for hypothesis in final_hypotheses:
            if hypothesis is None:
                errors_translation.append(0)
                errors_rotation.append(0)
                errors_ssd.append(0)
                errors_adi.append(0)
                errors_vsd.append(0)
                continue

            obj_id = int(hypothesis.model)
            obj_gt_T = obj_gt_Ts[obj_id]
            obj_est_T = hypothesis.transformation

            # 1) mean rot, tra (Mitash)
            err_t = np.linalg.norm(obj_est_T[:3, 3] - obj_gt_T[:3, 3]) * 1000  # [mm]

            symInfo = symInfos[obj_id]
            rotdiff = obj_est_T[:3, :3].I * obj_gt_T[:3, :3]
            rotErrXYZ = np.abs(Rotation.from_dcm(rotdiff).as_euler('XYZ', degrees=True))
            for dim in range(3):
                if symInfo[dim] == 90:
                    rotErrXYZ[dim] = abs(rotErrXYZ[dim] - 90)
                    rotErrXYZ[dim] = min(rotErrXYZ[dim], 90 - rotErrXYZ[dim])
                elif symInfo[dim] == 180:
                    rotErrXYZ[dim] = min(rotErrXYZ[dim], 180 - rotErrXYZ[dim])
                elif symInfo[dim] == 360:
                    rotErrXYZ[dim] = 0
            err_r = np.mean(rotErrXYZ)

            errs_t.append(err_t)
            errs_r.append(err_r)
            errors_translation.append(err_t)
            errors_rotation.append(err_r)

            if ALL_COSTS:
                # 2) SSD + ADI
                model_points = dataset.pcd[obj_id-1]

                def sym_steps(nfo):
                    if nfo == 90:
                        n = 4
                        d = np.deg2rad(90)
                    elif nfo == 180:
                        n = 2
                        d = np.deg2rad(180)
                    elif nfo == 360:
                        n = 315
                        d = 2.0 * np.pi / n
                    else:
                        n = 1
                        d = 0
                    return n, d

                if np.sum(symInfo) == 360*3:
                    err_ssd = cost_ADD(model_points, obj_gt_T, obj_est_T, symmetric=True)*1000  # to mm
                else:
                    xn, dx = sym_steps(symInfo[0])
                    yn, dy = sym_steps(symInfo[1])
                    zn, dz = sym_steps(symInfo[2])
                    Ts_sym = []
                    for x in range(xn):
                        for y in range(yn):
                            for z in range(zn):
                                # print(np.rad2deg([x * dx, y * dy, z * dz]))
                                Ts_sym.append(Rotation.from_euler('xyz', [x * dx, y * dy, z * dz]).as_dcm())

                    # print(len(Ts_sym))
                    ssds = []
                    for T_sym in Ts_sym:
                        obj_est_T_sym = obj_est_T.copy()
                        obj_est_T_sym[:3, :3] = np.matrix(obj_est_T_sym[:3, :3]) * np.matrix(T_sym)
                        ssds.append(cost_ADD(model_points, obj_gt_T, obj_est_T_sym, symmetric=False) * 1000)  # to mm
                    err_ssd = np.min(ssds)
                err_adi = cost_ADD(model_points, obj_gt_T, obj_est_T, symmetric=True)*1000  # to mm

                errs_ssd.append(err_ssd)
                errs_adi.append(err_adi)
                errors_ssd.append(err_ssd)
                errors_adi.append(err_adi)

                # 3) VSD
                observation = {
                    "depth": scene_depth,
                    "normals": scene_normals,
                    "extrinsics": camera_extrinsics,
                    "intrinsics": camera_intrinsics
                }
                gt_depth = renderer.render([obj_id], [obj_gt_T], camera_extrinsics, camera_intrinsics, mode='depth')[1]
                est_depth = hypothesis.render(observation, 'depth')[1]
                err_vsd, gt_visibility = cost_VSD(full_depth, gt_depth*1000, est_depth*1000)

                if gt_visibility > 0.1:  # TODO is this the same threshold as in Hodan?
                    errs_vsd.append(err_vsd)
                    errors_vsd.append(err_vsd)

        print("   mean err r = %0.1f" % np.mean(errs_r))
        print("   mean err t [cm] = %0.1f" % (np.mean(errs_t)/10))
        if ALL_COSTS:
            print("   ---")
            print("   mean SSD [mm] = %0.1f" % np.mean(errs_ssd))
            print("   mean ADI [mm] = %0.1f" % np.mean(errs_adi))
            print("   mean VSD = %0.1f" % np.mean(errs_vsd))
        # print("   refinement iterations = %i" % ref.ref_count)  # TODO
        #
        # --- vis
        if PLOT:
            observation = {
                "depth": scene_depth,
                "normals": scene_normals,
                "extrinsics": camera_extrinsics,
                "intrinsics": camera_intrinsics
            }

            vis = np.dstack((depth, depth, depth))/1000#
            vis = rgb.copy()
            rgb_ren = []
            for hypothesis in final_hypotheses:
                if hypothesis is not None:
                    rendered = hypothesis.render(observation, 'color')
                    rgb_ren.append(rendered[0])
                    # vis[rendered[0] != 0] = vis[rendered[0] != 0] * 0.3 + rendered[0][rendered[0] != 0]/255 * 0.7
                    vis[rendered[0] != 0] = vis[rendered[0] != 0] * 0.3 + rendered[0][rendered[0] != 0] * 0.7
            vis2 = rgb.copy()
            for obj_hypotheses in hypotheses:
                rendered = obj_hypotheses[0].render(observation, 'color')
                vis2[rendered[0] != 0] = vis2[rendered[0] != 0] * 0.3 + rendered[0][rendered[0] != 0] * 0.7

            obj_ids = [renderer.dataset.objlist.index(int(h.id[:2])) for h in
                       final_hypotheses if h is not None]  # TODO do this conversion in renderer
            obj_trafos = [h.transformation for h in final_hypotheses if h is not None]
            # TODO just pass a list of hypotheses alternatively

            # # a) render depth, compute score on CPU
            vis3 = depth - renderer.render(obj_ids, obj_trafos,
                                       camera_extrinsics, camera_intrinsics,
                                       mode='depth')[1]*1000
            vis3[scene_depth == 0] = 0
            # vis3[vis3 < -20] = 0
            # vis3[vis3 > 20] = 0

            def debug_draw():
                plt.subplot(2, 2, 2)
                plt.imshow(vis)
                plt.subplot(2, 2, 1)
                # plt.imshow(vis2)
                plt.imshow(np.array(PIL.Image.open(PATH_APC + "scene-%0.4d/debug_search/renderFinalClass.png" % scene)))
                plt.subplot(2, 2, 3)
                # plt.imshow(depth/1000*255 + rgb[:,:,0])
                plt.imshow((scene_normals+1)/2)
                # plt.imshow(unexplained)
                plt.subplot(2, 2, 4)
                plt.imshow(np.abs(vis3), vmin=0, vmax=20)
                plt.title("mean abs depth error\n = %0.3f" % np.mean(np.abs(vis3[scene_depth>0])))
            debug_draw()
            plt.show()
            # drawnow(debug_draw)
            # plt.pause(0.05)
            # plt.pause(1.0)

        print("   ---")
        print("   ~ icp/call=%0.1fms" % (np.mean(TrimmedIcp.icp_durations) * 1000))
        if len(renderer.runtimes) > 0:#MODE not in ["BASE", "PIR"]:
            print("   ~ rendering/call=%0.1fms" % (np.mean(np.sum(renderer.runtimes, axis=1)) * 1000))
            print("   ~ cost/call=%0.1fms" % (np.mean(Verefine.cost_durations) * 1000))
        print("   ---")
        print("   ~ icp/frame=%ims" % (np.sum(TrimmedIcp.icp_durations) * 1000))
        if len(renderer.runtimes) > 0:#MODE not in ["BASE", "PIR"]:
            print("   ~ rendering/frame=%ims" % (np.sum(np.sum(renderer.runtimes, axis=1)) * 1000))
            print("   ~ cost/frame=%ims" % (np.sum(Verefine.cost_durations) * 1000))

        TrimmedIcp.icp_durations.clear()
        renderer.runtimes.clear()
        Verefine.cost_durations.clear()

    # # convert to BOP format
    # with open("/mnt/Data/datasets/BOP19/exapc/test/000001/scene_camera.json", 'w') as file:
    #     json.dump(scene_camera, file)
    #
    # renderer.set_observation(depth.reshape(480, 640, 1), scene_normals.reshape(480, 640, 3))
    # scene_gt = {}
    # scene_gt_info = {}
    # for scene in range(1, 43):
    #
    #     ids = [renderer.dataset.objlist.index(id) for id in obj_ids[(scene - 1) * 3:scene * 3]]
    #     keys = ["%i-%i" % (scene, obj_id) for obj_id in obj_ids[(scene - 1) * 3:scene * 3]]
    #
    #     scene_depth = renderer.render(ids,
    #                                   [obj_gt_Ts[k] for k in keys],
    #                                   camera_extrinsics, camera_intrinsics, mode='depth')[1]
    #     scene_gt_objs = []
    #     scene_gt_infos = []
    #     for oi, (obj_id, k) in enumerate(zip(obj_ids[(scene-1)*3:scene*3], keys)):
    #         id = renderer.dataset.objlist.index(obj_id)
    #         obj_depth = renderer.render([id], [obj_gt_Ts[k]], camera_extrinsics, camera_intrinsics, mode='depth')[1]
    #
    #         mask = ((obj_depth > 0) * 255).astype(np.uint8)
    #         mask_ids = np.argwhere(mask > 0)
    #         bbox = [np.min(mask_ids[:, 0]), np.min(mask_ids[:, 1]),
    #                 np.max(mask_ids[:, 0]), np.max(mask_ids[:, 1])]
    #         bbox = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]  # x, y, width, height
    #         px_count_all = (mask > 0).sum()
    #         px_count_valid = np.logical_and(mask > 0, depth > 0).sum()
    #
    #         PIL.Image.fromarray(mask).save(
    #             "/mnt/Data/datasets/BOP19/exapc/test/000001/mask/%0.6d_%0.6d.png" % (scene, oi))
    #
    #         mask_visib = (np.logical_and(obj_depth > 0, obj_depth <= scene_depth) * 255).astype(np.uint8)
    #         mask_visib_ids = np.argwhere(mask_visib > 0)
    #         bbox_visib = [np.min(mask_visib_ids[:, 0]), np.min(mask_visib_ids[:, 1]),
    #                       np.max(mask_visib_ids[:, 0]), np.max(mask_visib_ids[:, 1])]
    #         bbox_visib = [bbox_visib[1], bbox_visib[0], bbox_visib[3]-bbox_visib[1], bbox_visib[2]-bbox_visib[0]]  # x, y, width, height
    #         px_count_visib = (mask_visib > 0).sum()
    #         visib_fract = px_count_visib / px_count_all
    #
    #         PIL.Image.fromarray(mask_visib).save(
    #             "/mnt/Data/datasets/BOP19/exapc/test/000001/mask_visib/%0.6d_%0.6d.png" % (scene, oi))
    #
    #         scene_gt_obj = {
    #             "cam_R_m2c": list(np.array(obj_gt_Ts[k][:3, :3]).reshape(9)),
    #             "cam_t_m2c": list(np.array(obj_gt_Ts[k][:3, 3]).reshape(3)*1000),
    #             "obj_id": obj_id
    #         }
    #         scene_gt_objs.append(scene_gt_obj)
    #
    #         scene_gt_info_obj = {
    #             "bbox_obj": [int(v) for v in bbox],
    #             "bbox_visib": [int(v) for v in bbox_visib],
    #             "px_count_all": int(px_count_all),
    #             "px_count_valid": int(px_count_valid),
    #             "px_count_visib": int(px_count_visib),
    #             "visib_fract": float(visib_fract)
    #         }
    #         scene_gt_infos.append(scene_gt_info_obj)
    #
    #     scene_gt[scene] = scene_gt_objs
    #     scene_gt_info[scene] = scene_gt_infos
    #
    # with open("/mnt/Data/datasets/BOP19/exapc/test/000001/scene_gt.json", 'w') as file:
    #     json.dump(scene_gt, file)
    #
    # with open("/mnt/Data/datasets/BOP19/exapc/test/000001/scene_gt_info.json", 'w') as file:
    #     json.dump(scene_gt_info, file)
    #
    # test_targets = []
    # for scene in range(1, 43):
    #     test_targets += [{"im_id": scene, "inst_count": 1, "obj_id": obj_id, "scene_id": 1}
    #                      for obj_id in obj_ids[(scene - 1) * 3:scene * 3]]
    # with open("/mnt/Data/datasets/BOP19/exapc/test_targets_bop19.json", 'w') as file:
    #     json.dump(test_targets, file)


    print("\n\n")
    print("total = %0.1fms" % (np.mean(durations)*1000))
    print("total (w/o first) = %0.1fms" % (np.mean(durations[1:])*1000))
    print("------")
    print("1-obj err r [deg] = %0.1f" % np.mean(np.array(errors_rotation)[ind_1]))
    print("1-obj err t [cm] = %0.1f" % (np.mean(np.array(errors_translation)[ind_1]) / 10))
    print("2-obj err r [deg] = %0.1f" % np.mean(np.array(errors_rotation)[ind_2]))
    print("2-obj err t [cm] = %0.1f" % (np.mean(np.array(errors_translation)[ind_2]) / 10))
    if len(errors_translation) > 90:
        print("3-obj err r [deg] = %0.1f" % np.mean(np.array(errors_rotation)[ind_3]))
        print("3-obj err t [cm] = %0.1f" % (np.mean(np.array(errors_translation)[ind_3]) / 10))
    print("ALL err r [deg] = %0.1f" % np.mean(errors_rotation))
    print("ALL err t [cm] = %0.1f" % (np.mean(errors_translation)/10))
    print("------")
    if ALL_COSTS:
        for ii, sub_indices in enumerate([ind_1, ind_2, ind_3]):
            print("-- %i:" % (ii+1))
            print("SSD <1cm = %0.1f%%" % (np.mean(np.array(errors_ssd)[sub_indices] < 10) * 100))
            print("SSD <2cm = %0.1f%%" % (np.mean(np.array(errors_ssd)[sub_indices] < 20) * 100))
            print("ADI <1cm = %0.1f%%" % (np.mean(np.array(errors_adi)[sub_indices] < 10) * 100))
            print("ADI <2cm = %0.1f%%" % (np.mean(np.array(errors_adi)[sub_indices] < 20) * 100))
            print("------")
            print("mAP SSD <1cm = %0.1f%%" % (compute_mAP(np.array(errors_ssd)[sub_indices], 10) * 100))
            print("mAP SSD <2cm = %0.1f%%" % (compute_mAP(np.array(errors_ssd)[sub_indices], 20) * 100))
            print("mAP ADI <1cm = %0.1f%%" % (compute_mAP(np.array(errors_adi)[sub_indices], 10) * 100))
            print("mAP ADI <2cm = %0.1f%%" % (compute_mAP(np.array(errors_adi)[sub_indices], 20) * 100))
            print("------")
            print("VSD <0.3 = %0.1f%%" % (np.mean(np.array(errors_vsd)[sub_indices] < 0.3) * 100))
            print("------")
        print("-- all:")
        print("SSD <1cm = %0.1f%%" % (np.mean(np.array(errors_ssd) < 10) * 100))
        print("SSD <2cm = %0.1f%%" % (np.mean(np.array(errors_ssd) < 20) * 100))
        print("ADI <1cm = %0.1f%%" % (np.mean(np.array(errors_adi) < 10) * 100))
        print("ADI <2cm = %0.1f%%" % (np.mean(np.array(errors_adi) < 20) * 100))
        print("------")
        print("mAP SSD <1cm = %0.1f%%" % (compute_mAP(np.array(errors_ssd), 10) * 100))
        print("mAP SSD <2cm = %0.1f%%" % (compute_mAP(np.array(errors_ssd), 20) * 100))
        print("mAP ADI <1cm = %0.1f%%" % (compute_mAP(np.array(errors_adi), 10) * 100))
        print("mAP ADI <2cm = %0.1f%%" % (compute_mAP(np.array(errors_adi), 20) * 100))
        print("------")
        print("VSD <0.3 = %0.1f%%" % (np.mean(np.array(errors_vsd) < 0.3) * 100))
        print("------")

    # print("setup = %0.1fms" % (np.mean(simulator.runtimes, axis=0)[0]*1000))
    # print("sim   = %0.1fms" % (np.mean(simulator.runtimes, axis=0)[1]*1000))
    # print("read  = %0.1fms" % (np.mean(simulator.runtimes, axis=0)[2]*1000))
    if MODE in ["SV", "VF"]:
        print("sel = %0.1fms" % (np.mean(Verefine.duration_select) * 1000))
        print("exp = %0.1fms" % (np.mean(Verefine.duration_expand) * 1000))
        print("rol = %0.1fms" % (np.mean(Verefine.duration_rollout) * 1000))
        print("bac = %0.1fms" % (np.mean(Verefine.duration_backprop) * 1000))
    if MODE != "BASE":
        simulator.deinitialize()
