import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.io as scio
from scipy.spatial.transform.rotation import Rotation
import os
import PIL
import time
import torch
import gc
import random
import sys
sys.path.append("/home/dominik/projects/hsr-grasping")
sys.path.append("/home/dominik/projects/hsr-grasping/src")
sys.path.append("/home/dominik/projects/hsr-grasping/src/util")

# make reproducible (works up to BAB -- TODO VF smh not)
seed = 0
torch.manual_seed(seed)  # cpu
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)

from src.util.fast_renderer import Renderer
from src.util.dataset import LmDataset
from src.densefusion.densefusion import DenseFusion
from src.verefine.simulator import Simulator
from src.verefine.verefine import Hypothesis, PhysIR, BudgetAllocationBandit
import src.verefine.verefine as Verefine
from src.icp.icp import TrimmedIcp
from src.verefine.plane_segmentation import PlaneDetector


# settings
PATH_BOP19 = "/mnt/Data/datasets/BOP19/"
PATH_LM = "/mnt/Data/datasets/SIXD/LM_LM-O/"
PATH_LM_ROOT = '/mnt/Data/datasets/Linemod_preprocessed/'

MODE = "BAB"  # "BASE", "PIR", "BAB", "SV", "VF", MITASH
EST_MODE = "GT"  # "GT", "DF", "PCS"
REF_MODE = "DF"  # "DF", "ICP"
SEGMENTATION = "GT"  # GT, PCNN
EXTRINSICS = "GT"  # GT, PLANE
TAU = 20
TAU_VIS = 10  # [mm]

# BREAK IT
# 1) initial pose
MODE_OFF = "r"  # "t", "r"
OFFSETS = [5]#[5, 10, 15, 20, 30, 35, 40, 45, 50]#[0, 25, 50]#list(range(0, 51, 10))  # TODO list(range(0, 51, 5))#list(range(0, 91, 10)) if MODE_OFF == "r" else list(range(0, 51, 5))
# 2) depth noise
MODE_NOISE = "sample"  # sample, patch, ""
NOISE_SAMPLE_P = 0.4  # percentage of samples to be removed in mode "sample"
NOISE_PATCH_P = 0.4  # percentage of the bbox size to use for the occlusion patch in mode "patch"
NOISE_TOP_P = 0.4  # the top x% (w.r.t. object height) are cut-off in mode "top"


SCENES = []

PLOT = False
USE_NORMALS = False

obj_names = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone"
}

# -----------------

if __name__ == "__main__":

    dataset = LmDataset(base_path=PATH_LM)
    SCENES = dataset.objlist[1:]

    if len(sys.argv) > 1 and len(sys.argv) >= 3:
        # refine mode, offset mode and offset size (0 is script path)
        MODE = sys.argv[1]
        MODE_OFF = sys.argv[2]
        # OFFSETS = [int(sys.argv[3])]
        SCENES = dataset.objlist[1:] if len(sys.argv) == 3 else [int(sys.argv[3])]
        PLOT = False
    print("mode: %s -- offsets: %s %s -- scenes: %s" % (MODE, MODE_OFF, OFFSETS, SCENES))

    df = None
    durations = []

    # if MODE != "BASE" or REF_MODE == "ICP":
    renderer = Renderer(dataset, recompute_normals=USE_NORMALS)
    Verefine.RENDERER = renderer
        # renderer.create_egl_context()

    if MODE != "BASE":
        simulator = Simulator(dataset, instances_per_object=Verefine.HYPOTHESES_PER_OBJECT)
        Verefine.SIMULATOR = simulator
        pir = None


    # # get all keyframes
    # with open(PATH_LM + "image_sets/keyframe.txt", 'r') as file:
    #     keyframes = file.readlines()
    # keyframes = [keyframe.replace("\n", "") for keyframe in keyframes]

    # get all test targets
    with open(PATH_BOP19 + "lm/test_targets_bop19.json", 'r') as file:
        targets = json.load(file)

    im_ids, scene_ids, inst_counts, obj_ids = [], [], [], []
    for target in targets:
        im_ids.append(target['im_id'])
        scene_ids.append(target['scene_id'])
        inst_counts.append(target['inst_count'])
        obj_ids.append(target['obj_id'])
    im_ids, scene_ids, obj_ids = np.array(im_ids), np.array(scene_ids), np.array(obj_ids)

    # loop over scenes...
    objects = []
    for scene in SCENES:
        print("scene %i..." % scene)

        class_id = dataset.objlist.index(scene)

        scene_target_indices = np.argwhere(scene_ids == scene)

        # frames and objects in scene
        scene_im_ids = im_ids[scene_target_indices]
        scene_obj_ids = obj_ids[scene_target_indices]

        # camera infos
        with open(PATH_BOP19 + "lm/test/%0.6d/scene_camera.json" % scene, 'r') as file:
            scene_camera = json.load(file)

        # scene gt
        with open(PATH_BOP19 + "lm/test/%0.6d/scene_gt.json" % scene, 'r') as file:
            scene_gt = json.load(file)
        with open(PATH_BOP19 + "lm/test/%0.6d/scene_gt_info.json" % scene, 'r') as file:
            scene_gt_info = json.load(file)

        # loop over frames in scene...
        frames = sorted(np.unique(scene_im_ids))
        for fi, frame in enumerate(frames):
            print("   frame %i (%i/%i)..." % (frame, fi+1, len(frames)))

            # objects in frame
            frame_target_indices = np.argwhere(scene_im_ids == frame)[:, 0]

            # # meta data (PCNN segmentation and poses)
            # keyframe = keyframes.index("%0.4d/%0.6d" % (scene, frame))
            # meta = scio.loadmat(PATH_LM + "../YCB_Video_toolbox/results_PoseCNN_RSS2018/%0.6d.mat" % keyframe)

            # load observation
            rgb = np.array(PIL.Image.open(PATH_BOP19 + "lm/test/%0.6d/rgb/%0.6d.png" % (scene, frame)))
            depth = np.array(PIL.Image.open(PATH_BOP19 + "lm/test/%0.6d/depth/%0.6d.png" % (scene, frame)))

            if SEGMENTATION == "GT":
                labels = np.array(PIL.Image.open(PATH_BOP19 + "lm/test/%0.6d/mask/%0.6d_000000.png" % (scene, frame)))
            elif SEGMENTATION == "PCNN":
                labels = np.array(PIL.Image.open(PATH_LM_ROOT + "segnet_results/%0.2d_label/%0.4d_label.png" % (scene, frame)))
            else:
                raise ValueError("SEGMENTATION can only be GT or PCNN")

            # get obj ids, num objs, obj names
            frame_obj_ids = [scene]  #scene_obj_ids[frame_target_indices] if SEGMENTATION == "GT" else np.unique(labels)[1:]
            frame_num_objs = 1  # len(frame_obj_ids)
            frame_obj_names = [obj_names[int(idx)] for idx in frame_obj_ids]

            # get extrinsics and intrinsics
            frame_camera = scene_camera[str(frame)]

            camera_intrinsics = np.array(frame_camera["cam_K"]).reshape(3, 3)

            if df is None:
                if EST_MODE == "DF" or REF_MODE == "DF":
                    df = DenseFusion(640, 480, camera_intrinsics, dataset, mode="bab")  # TODO set this correctly or doesn't matter non-ros refine?

                if REF_MODE == "ICP":
                    ref = TrimmedIcp(renderer, camera_intrinsics, dataset, mode="bab")
                else:
                    ref = df

                if MODE != "BASE":
                    pir = PhysIR(ref, simulator)
                    Verefine.REFINER = pir

            # get pose estimates
            obj_ids_ = [scene]  #[int(v) for v in meta['rois'][:, 1]]
            pose_info = scene_gt['%i' % frame][0]
            pose = np.matrix(np.eye(4))
            pose[:3, :3] = np.array(pose_info['cam_R_m2c']).reshape(3, 3)
            pose[:3, 3] = np.array(pose_info['cam_t_m2c']).reshape(3, 1)/1000
            obj_poses = [pose]

            if EXTRINSICS == "PLANE":
                plane_detector = PlaneDetector(640, 480, camera_intrinsics, down_scale=8)

                # camera_extrinsics = np.matrix(np.eye(4))
                # camera_extrinsics[:3, :3] = np.matrix(frame_camera["cam_R_w2c"]).reshape(3, 3)
                # camera_extrinsics[:3, 3] = np.matrix(frame_camera["cam_t_w2c"]).T / 1000.0
                try:
                    camera_extrinsics = plane_detector.detect(depth, labels == 255)
                except ValueError as ex:
                    print(ex)
                    print("skipping frame...")
                    continue
            elif EXTRINSICS == "GT":
                camera_extrinsics = pose.copy()
                camera_extrinsics[:3, 3] = camera_extrinsics[:3, 3] + camera_extrinsics[:3, :3]*np.matrix([0.0, 0.0, dataset.obj_bot[scene-1]]).T
            else:
                raise ValueError("EXTRINSICS needs to be GT (from model) or PLANE (segmentation).")

            # TODO still a bit leaky -- check using fast.ai GPUMemTrace
            gc.collect()
            torch.cuda.empty_cache()

            obj_mask = labels == 255
            mask_ids = np.argwhere(labels == 255)
            obj_roi = scene_gt_info['%i' % frame][0]['bbox_obj']

            scene_depth = depth.copy()
            scene_depth[obj_mask == 0] = 0

            # -- noisy depth
            if MODE_NOISE == "sample":
                # a) robustness - randomly remove n samples
                num_ids = len(mask_ids)
                noise_ids = mask_ids[np.random.choice(list(range(num_ids)), int(NOISE_SAMPLE_P * num_ids)), :]
                scene_depth[noise_ids[:, 0], noise_ids[:, 1]] = 0
            elif MODE_NOISE == "patch":
                # b) robustness - remove a contiguous blob, greedily grown from a random position on obj_mask
                num_ids = len(mask_ids)
                patch_center = mask_ids[np.random.randint(0, num_ids), :]


                def grow(blob, cur_center, total_count, max_count):
                    count = 0
                    frontier = []
                    pattern = []
                    max_size = 1  # 1 for small square, 2 + if for circle
                    for square_size in range(max_size + 1):
                        for u in range(-square_size, square_size + 1):  # square
                            for v in range(-square_size, square_size + 1):
                                # if np.abs(u) == np.abs(
                                #         v) and square_size == max_size:  # if max_size>1, makes it a circle
                                #     continue
                                pattern.append([u, v])
                    # pattern = [[0, -1], [-1, 0], [1, 0], [0, 1]]  # small star
                    for u, v in pattern:
                        if total_count + count + 1 > max_count:
                            break
                        new_center = cur_center + [u, v]
                        if (blob[new_center[0], new_center[1]] == 1 or
                                obj_mask[new_center[0], new_center[1]] == 0):  # known or background
                            continue
                        blob[new_center[0], new_center[1]] = 1
                        count += 1
                        frontier.append(new_center)
                    return count, frontier


                def grow_frontier(blob, count, max_count, frontier):
                    while count < max_count:
                        new_frontier = []
                        for new_center in frontier:
                            if count < max_count:
                                grow_count, grow_frontier = grow(blob, new_center, count, max_count)
                                count += grow_count
                                new_frontier += grow_frontier
                        frontier = new_frontier.copy()
                    return count, frontier


                blob = np.zeros_like(obj_mask)
                count = 0
                frontier = [patch_center]
                count, frontier = grow_frontier(blob, count=count, max_count=int(NOISE_SAMPLE_P * num_ids), frontier=frontier)
                # plt.imshow(blob)
                # plt.plot(patch_center[1], patch_center[0], 'rx')
                # plt.title(str((blob > 0).sum()))

                scene_depth[blob > 0] = 0
            elif MODE_NOISE == "top":
                # c) bias - remove same part of object -> check whether this "pushes" results in a direction (bias)
                obs = depth.reshape(480, 640, 1)
                #     obs[labels == 0] = 0  # makes results worse (with PCNN seg)
                renderer.set_observation(obs)
                cutoff_plane = camera_extrinsics.copy()
                cutoff_plane[:3, 3] -= 2 * (
                            camera_extrinsics[:3, :3] * np.matrix([0.0, 0.0, dataset.obj_bot[scene - 1]]).T) * (1-NOISE_TOP_P)
                cutoff_mask = renderer.render([0], [cutoff_plane], camera_extrinsics, camera_intrinsics,
                                                          mode='depth')[1] * 1000 >= depth
                # plt.imshow(np.logical_and(obj_mask, np.logical_not(cutoff_mask)))
                scene_depth[cutoff_mask > 0] = 0
                # plt.imshow(scene_depth)
                # plt.show()
            # else: "" -> no artificial depth signal noise


            def estimate_normals(D):
                D_px = D.copy() * camera_intrinsics[0, 0]  # from meters to pixels
                # # Sobel
                # dzdx = cv.Sobel(depth, cv.CV_64F, dx=1, dy=0, ksize=-1)  # difference
                # dzdy = cv.Sobel(depth, cv.CV_64F, dx=0, dy=1, ksize=-1)  # step size of 1px
                import cv2 as cv
                # Prewitt
                kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                dzdx = cv.filter2D(D_px, -1, kernelx)
                dzdy = cv.filter2D(D_px, -1, kernely)
                normal = np.dstack((-dzdx, -dzdy, D_px != 0.0))  # only where we have a depth value
                n = np.linalg.norm(normal, axis=2)
                n = np.dstack((n, n, n))
                normal = np.divide(normal, n, where=(n != 0))
                normal[n == 0] = 0.0
                # plt.imshow((normal + 1) / 2)
                # plt.show()
                return normal

            observation = {
                "color": rgb,
                "depth": scene_depth,  # TODO or depth?
                "normals": estimate_normals(scene_depth/1000),
                "extrinsics": camera_extrinsics,
                "intrinsics": camera_intrinsics
            }

            def get_bbox(posecnn_rois):
                border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640,
                               680]
                rmin = int(posecnn_rois[3]) + 1
                rmax = int(posecnn_rois[5]) - 1
                cmin = int(posecnn_rois[2]) + 1
                cmax = int(posecnn_rois[4]) - 1
                r_b = rmax - rmin
                for tt in range(len(border_list)):
                    if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                        r_b = border_list[tt + 1]
                        break
                c_b = cmax - cmin
                for tt in range(len(border_list)):
                    if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                        c_b = border_list[tt + 1]
                        break
                center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
                rmin = center[0] - int(r_b / 2)
                rmax = center[0] + int(r_b / 2)
                cmin = center[1] - int(c_b / 2)
                cmax = center[1] + int(c_b / 2)
                if rmin < 0:
                    delt = -rmin
                    rmin = 0
                    rmax += delt
                if cmin < 0:
                    delt = -cmin
                    cmin = 0
                    cmax += delt
                if rmax > 480:
                    delt = rmax - 480
                    rmax = 480
                    rmin -= delt
                if cmax > 640:
                    delt = cmax - 640
                    cmax = 640
                    cmin -= delt
                return rmin, rmax, cmin, cmax


            obj_roi = [0, 0, obj_roi[0], obj_roi[1], obj_roi[0] + obj_roi[2], obj_roi[1] + obj_roi[3]]
            vmin, vmax, umin, umax = get_bbox(obj_roi)
            obj_roi = [vmin, umin, vmax, umax]

            # hypotheses.append(Hypothesis("%0.2d" % obj_id, obj_T, obj_roi, obj_mask, None, None, 0, obj_confidence))
            # estimate = [obj_q, obj_pose[4:].reshape(3, 1), obj_confidence]
            # _, _, _, emb, cloud = df.forward(rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id)  # TODO use this to generate hypotheses using DF

            if EST_MODE == "DF":
                estimates, emb, cloud = df.estimate(rgb, depth, camera_intrinsics, obj_roi, obj_mask, class_id,
                                                    Verefine.HYPOTHESES_PER_OBJECT)
            elif REF_MODE == "DF":
                _, _, _, emb, cloud = df.forward(rgb, depth, camera_intrinsics, obj_roi, obj_mask, class_id)
            else:
                emb, cloud = None, None

            for OFFSET in OFFSETS:
                print("      offset %i" % OFFSET)

                obj_Ts = []
                hypotheses = []
                offsets = []
                estimates = []
                refiner_params = []

                # TODO still a bit leaky -- check using fast.ai GPUMemTrace
                gc.collect()
                torch.cuda.empty_cache()

                if emb is not None or REF_MODE != "DF":
                    if EST_MODE == "DF":
                        # a) take n estimates
                        new_hypotheses = []
                        new_refiner_params = []
                        for hi, estimate in enumerate(estimates):
                            obj_T = np.matrix(np.eye(4))
                            obj_T[:3, :3] = Rotation.from_quat(estimate[0]).as_dcm()
                            obj_T[:3, 3] = estimate[1].reshape(3, 1)
                            obj_confidence = estimate[2]
                            refiner_param = [rgb, depth, camera_intrinsics, obj_roi, obj_mask, class_id, estimate,
                                             Verefine.ITERATIONS, emb, cloud, None]
                            new_hypotheses.append(Hypothesis("%0.2d" % scene, obj_T, obj_roi, obj_mask, None, None, hi,
                                                             obj_confidence, refiner_param=refiner_param))
                            # new_refiner_params.append([rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id, estimate, Verefine.ITERATIONS, emb, cloud])
                            if REF_MODE == "ICP":
                                refiner_params[-1][-3:-1] = None, None
                        hypotheses += [new_hypotheses]
                        refiner_params += [new_refiner_params]
                    elif EST_MODE == "GT":
                        # b) perturb GT
                        obj_T = pose
                        q = Rotation.from_dcm(obj_T[:3, :3]).as_quat()
                        t = obj_T[:3, 3]
                        c = 1.0
                        estimate = [q, t, c]

                        # hypotheses.append(
                        #     Hypothesis("%0.2d" % scene, obj_T.copy(), obj_roi, obj_mask, None, None, 0, c))
                        # refiner_params.append(
                        #     [rgb, depth, camera_intrinsics, obj_roi, obj_mask, scene, estimate, Verefine.ITERATIONS, emb,
                        #      cloud])

                        new_hypotheses = []
                        # 1) sample +- per axis
                        # for axis in ['x', 'y', 'z']:
                        #     for offset in [-OFFSET, OFFSET]:
                        #         hi = len(new_hypotheses)
                        #
                        #         if MODE_OFF == "r":
                        #             new_rotation = obj_T[:3, :3] * np.matrix(Rotation.from_euler(axis, offset, degrees=True).as_dcm())
                        #             new_translation = obj_T[:3, 3].copy()
                        #         else:
                        #             new_rotation = obj_T[:3, :3].copy()
                        #             new_translation = obj_T[:3, 3].copy()
                        #             new_translation[int(np.argwhere(np.array(['x', 'y', 'z']) == axis))] += offset/1000
                        #
                        #         new_estimate = [Rotation.from_dcm(new_rotation).as_quat(), new_translation, estimate[2]]
                        #         new_obj_T = np.matrix(np.eye(4))
                        #         new_obj_T[:3, :3] = new_rotation
                        #         new_obj_T[:3, 3] = new_translation
                        #
                        #         refiner_param = [rgb, depth, camera_intrinsics, obj_roi, obj_mask, scene, new_estimate,
                        #              Verefine.ITERATIONS, emb, cloud, None]
                        #         new_hypotheses.append(
                        #             Hypothesis("%0.2d" % scene, new_obj_T.copy(), obj_roi, obj_mask, None, None, hi,
                        #                         c, refiner_param=refiner_param))
                        # 2) sample from unit sphere
                        def sample_from_sphere(scale=1.0):
                            sample = np.random.randn(3, 1)
                            sample /= np.linalg.norm(sample, axis=0)
                            sample *= scale
                            return sample


                        new_offsets = []
                        for hi in range(Verefine.HYPOTHESES_PER_OBJECT):
                            if MODE_OFF == "r":
                                # quaternion as rotation by ANGLE_OFF about random unit axis (X, Y, Z)
                                # -> q = (X*s, Y*s, Z*s, c), where c,s are cos/sin of ANGLE_OFF/2
                                rotation_vector = sample_from_sphere()  # (X, Y, Z)
                                c, s = np.cos(np.deg2rad(OFFSET/2)), np.sin(np.deg2rad(OFFSET/2))
                                rotation_off = np.vstack((rotation_vector * s, [c])).T  # q

                                new_rotation = obj_T[:3, :3] * np.matrix(Rotation.from_quat(rotation_off).as_dcm())
                                new_translation = obj_T[:3, 3].copy()

                                new_offsets.append(rotation_off)
                            else:
                                translation_off = sample_from_sphere(scale=OFFSET/1000)

                                new_rotation = obj_T[:3, :3].copy()
                                new_translation = obj_T[:3, 3].copy() + translation_off

                                new_offsets.append(translation_off)

                            new_estimate = [Rotation.from_dcm(new_rotation).as_quat(), new_translation, estimate[2]]
                            new_obj_T = np.matrix(np.eye(4))
                            new_obj_T[:3, :3] = new_rotation
                            new_obj_T[:3, 3] = new_translation

                            refiner_param = [rgb, depth, camera_intrinsics, obj_roi, obj_mask, class_id, new_estimate,
                                 Verefine.ITERATIONS, emb, cloud, None]
                            new_hypotheses.append(
                                Hypothesis("%0.2d" % scene, new_obj_T.copy(), obj_roi, obj_mask, None, None, hi,
                                            c, refiner_param=refiner_param))

                        hypotheses += [new_hypotheses]
                        offsets += [new_offsets]

                # --- refine
                final_hypotheses = []
                st = time.time()

                if MODE != "BASE":
                    # init frame
                    simulator.initialize_frame(camera_extrinsics)
                # if MODE != "BASE" or REF_MODE == "ICP":
                obs = depth.reshape(480, 640, 1)  # TODO scene depth!
                #     obs[labels == 0] = 0  # makes results worse (with PCNN seg)
                renderer.set_observation(obs)

                # plt.figure()
                # # hi = 0
                # # plt_indices = [1, 4, 2, 5, 3, 6]
                # # for axis in ['x', 'y', 'z']:
                # #     for offset in [-OFFSET, OFFSET]:
                # #         plt.subplot(2, 3, plt_indices[hi])
                # #         plt.imshow(hypotheses[0][hi].render(observation, 'color')[0] / 255 * 0.7 + rgb / 255 * 0.3)
                # #         plt.title(axis + " %0.1f" % offset + ("mm" if MODE_OFF == "t" else "deg"))
                # #         hi += 1
                # for obj_offsets, obj_hypotheses in zip(offsets, hypotheses):
                #
                #     rows = np.floor(np.sqrt(len(obj_offsets)))
                #     cols = np.ceil(len(obj_offsets) / rows)
                #
                #     for hi, (offset, hypothesis) in enumerate(zip(obj_offsets, obj_hypotheses)):
                #         plt.subplot(rows, cols, hi+1)
                #         plt.imshow(hypothesis.render(observation, 'color')[0] / 255 * 0.7 + rgb / 255 * 0.3)
                #         plt.title(",".join(["%0.1f" % v for v in (offset*1000)]) + "mm (=%0.1f)"
                #                   % (np.linalg.norm(offset)*1000) if MODE_OFF == "t"
                #                   else ("%0.1f" % np.rad2deg(np.arccos(offset[0][-1])*2) + "deg about "
                #                         + ",".join(["%0.1f" % (v/np.sin(np.arccos(offset[0][-1]))) for v in offset[0][:-1]])))
                #         hi += 1
                # plt.show()

                if MODE == "BASE":
                    # DF (base)
                    refinements = 0
                    # refine only max conf
                    for obj_hypotheses in hypotheses:
                        hypothesis = obj_hypotheses[0]  # only take best

                        refinements += refiner_param[-4]
                        q, t, c = ref.refine(*hypothesis.refiner_param)

                        hypothesis.transformation[:3, :3] = Rotation.from_quat(q).as_dcm()
                        hypothesis.transformation[:3, 3] = t.reshape(3, 1)
                        hypothesis.confidence = c
                        final_hypotheses.append(hypothesis)
                    # print(refinements)

                elif MODE in ["PIR", "MITASH"]:
                    # PIR
                    for obj_hypotheses in hypotheses:
                        hypothesis = obj_hypotheses[0]  # only take best

                        if MODE == "PIR":
                            phys_hypotheses = pir.refine(hypothesis)
                            final_hypotheses.append(phys_hypotheses[-1])  # pick hypothesis after last refinement step
                        else:  # MITASH
                            final_hypotheses.append(pir.refine_mitash(hypothesis))

                elif MODE == "BAB":
                    # BAB (with PIR)
                    refinements = 0
                    for obj_hypotheses in hypotheses:
                        # set according to actual number of hypotheses (could be less for PCS if we don't find enough)
                        # Verefine.MAX_REFINEMENTS_PER_HYPOTHESIS = Verefine.ITERATIONS * Verefine.REFINEMENTS_PER_ITERATION * len(obj_hypotheses)

                        bab = BudgetAllocationBandit(pir, observation, obj_hypotheses)
                        bab.refine_max()

                        # # TODO debug plot
                        if PLOT:
                            for oi, hs in enumerate(bab.pool):
                                for hi, h in enumerate(hs):
                                    if h is None:
                                        continue
                                    plt.subplot(len(bab.pool), len(hs), oi * len(hs) + hi + 1)
                                    # ren_d = h.render(bab.observation, 'depth')[1] * 1000
                                    # vis = np.abs(ren_d - bab.observation['depth'])
                                    # vis[ren_d == 0] = 0
                                    ren_c = h.render(bab.observation, 'color')[0]
                                    vis = ren_c/255*0.7 + rgb/255*0.3
                                    plt.imshow(vis[h.roi[0]-50:h.roi[2]+50, h.roi[1]-50:h.roi[3]+50])
                                    plt.title("%0.2f" % bab.fits[oi, hi])
                            plt.show()

                        hypothesis, plays, fit = bab.get_best()
                        assert hypothesis is not None
                        final_hypotheses.append(hypothesis)

                        refinements += bab.max_iter - len(obj_hypotheses)  # don't count initial render
                    # print(refinements)

                elif MODE == "SV":
                    pass  # TODO this should be similar to mitash, just everything at scene level -> no BAB, adapt candidate generation

                elif MODE == "VF":
                    hypotheses_pool = dict()
                    for obj_hypotheses in hypotheses:
                        hypotheses_pool[obj_hypotheses[0].model] = obj_hypotheses

                    Verefine.OBSERVATION = observation
                    final_hypotheses, final_fit, convergence_hypotheses = Verefine.verefine_solution(hypotheses_pool)

                    # if Verefine.TRACK_CONVERGENCE:
                    #     # # fill-up to 200 with final hypothesis
                    #     # convergence_hypotheses += [final_hypotheses] * (200 - len(convergence_hypotheses))
                    #
                    #     # write results
                    #     for convergence_iteration, iteration_hypotheses in convergence_hypotheses.items():
                    #         iteration_hypotheses = [hs[0] for hs in iteration_hypotheses]
                    #         for hypothesis in iteration_hypotheses:
                    #             with open("/home/dominik/projects/hsr-grasping/convergence-vf5-c1/%0.3d_ycbv-test.csv"
                    #                       % convergence_iteration, 'a') as file:
                    #                 parts = ["%0.2d" % scene, "%i" % frame, "%i" % int(hypothesis.model),
                    #                          "%0.3f" % hypothesis.confidence,
                    #                          " ".join(
                    #                              ["%0.6f" % v for v in np.array(hypothesis.transformation[:3, :3]).reshape(9)]),
                    #                          " ".join(
                    #                              ["%0.6f" % (v * 1000) for v in np.array(hypothesis.transformation[:3, 3])]),
                    #                          "%0.3f" % 1.0]
                    #                 file.write(",".join(parts) + "\n")

                    final_hypotheses = [hs[0] for hs in final_hypotheses]

                durations.append(time.time() - st)

                # plt.figure()
                # plt.subplot(1, 2, 1)
                # plt.imshow(renderer.render([scene], [pose], camera_extrinsics, camera_intrinsics,
                #                            mode='color')[0]/255*0.7 + rgb/255*0.3)
                # plt.title("GT")
                # plt.subplot(1, 2, 2)
                # plt.imshow(final_hypotheses[0].render(observation, 'color')[0]/255*0.7 + rgb/255*0.3)
                # plt.title("refined")
                # plt.show()

                # # write results
                if not PLOT:
                    for hypothesis in final_hypotheses:
                        # with open("/home/dominik/projects/hsr-grasping/break/GT_lm-test.csv",
                        #           'a') as file:
                        # with open("/home/dominik/projects/hsr-grasping/break/%s%s%s%0.2d_lm-test.csv"
                        with open("/home/dominik/projects/hsr-grasping/log/%s/%s%s%0.2d_lm-test.csv"
                                  % (MODE, "%i-" % Verefine.HYPOTHESES_PER_OBJECT if MODE=="BAB" else "", MODE_OFF, OFFSET), 'a') as file:
                            parts = ["%0.2d" % scene, "%i" % frame, "%i" % int(hypothesis.model), "%0.3f" % hypothesis.confidence,
                                     " ".join(["%0.6f" % v for v in np.array(hypothesis.transformation[:3, :3]).reshape(9)]),
                                     " ".join(["%0.6f" % (v * 1000) for v in np.array(hypothesis.transformation[:3, 3])]),
                                     "%0.3f" % durations[-1]]
                            file.write(",".join(parts) + "\n")
            # break
        # break

    print("total = %0.1fms" % (np.mean(durations)*1000))
    print("total (w/o first) = %0.1fms" % (np.mean(durations[1:])*1000))
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
