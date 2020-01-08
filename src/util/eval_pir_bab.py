# import matplotlib
# matplotlib.use("Qt5Agg")
# import matplotlib.pyplot as plt
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

# make reproducible
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
from src.util.dataset import YcbvDataset
from src.densefusion.densefusion import DenseFusion
from src.verefine.simulator import Simulator
from src.verefine.verefine import Hypothesis, PhysIR, BudgetAllocationBandit
import src.verefine.verefine as Verefine
from src.icp.icp import TrimmedIcp


# settings
PATH_BOP19 = "/mnt/Data/datasets/BOP19/"
PATH_YCBV = "/mnt/Data/datasets/YCB Video/YCB_Video_Dataset/"

MODE = "VF"  # "BASE", "PIR", "BAB", "SV", "VF"
EST_MODE = "DF"  # "DF", "PCS"
REF_MODE = "DF"  # "DF", "ICP"
SEGMENTATION = "PCNN"  # GT, PCNN
TAU = 20
TAU_VIS = 10  # [mm]

obj_names = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick"
}

# -----------------

if __name__ == "__main__":

    dataset = YcbvDataset(base_path=PATH_YCBV)

    df = None
    durations = []

    if MODE != "BASE" or REF_MODE == "ICP":
        renderer = Renderer(dataset)
        Verefine.RENDERER = renderer
        # renderer.create_egl_context()

    if MODE != "BASE":
        simulator = Simulator(dataset, instances_per_object=Verefine.HYPOTHESES_PER_OBJECT)#, objects_to_use=[1, 6, 9, 12, 14, 16, 17, 19, 20, 21])
        Verefine.SIMULATOR = simulator
        pir = None


    # get all keyframes
    with open(PATH_YCBV + "image_sets/keyframe.txt", 'r') as file:
        keyframes = file.readlines()
    keyframes = [keyframe.replace("\n", "") for keyframe in keyframes]

    # get all test targets
    with open(PATH_BOP19 + "ycbv/test_targets_bop19.json", 'r') as file:
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
    scenes = [48]  # sorted(np.unique(scene_ids))  #
    for scene in scenes:
        print("scene %i..." % scene)
        scene_target_indices = np.argwhere(scene_ids == scene)

        # frames and objects in scene
        scene_im_ids = im_ids[scene_target_indices]
        scene_obj_ids = obj_ids[scene_target_indices]

        # camera infos
        with open(PATH_BOP19 + "ycbv/test/%0.6d/scene_camera.json" % scene, 'r') as file:
            scene_camera = json.load(file)

        # loop over frames in scene...
        frames = sorted(np.unique(scene_im_ids))
        for fi, frame in enumerate(frames):
            print("   frame %i (%i/%i)..." % (frame, fi+1, len(frames)))

            # objects in frame
            frame_target_indices = np.argwhere(scene_im_ids == frame)[:, 0]

            # meta data (PCNN segmentation and poses)
            keyframe = keyframes.index("%0.4d/%0.6d" % (scene, frame))
            meta = scio.loadmat(PATH_YCBV + "../YCB_Video_toolbox/results_PoseCNN_RSS2018/%0.6d.mat" % keyframe)

            # load observation
            rgb = np.array(PIL.Image.open(PATH_BOP19 + "ycbv/test/%0.6d/rgb/%0.6d.png" % (scene, frame)))
            depth = np.array(PIL.Image.open(PATH_BOP19 + "ycbv/test/%0.6d/depth/%0.6d.png" % (scene, frame))) / 10.0

            if SEGMENTATION == "GT":
                labels = np.array(PIL.Image.open(PATH_YCBV + "data/%0.4d/%0.6d-label.png" % (scene, frame)))
            elif SEGMENTATION == "PCNN":
                labels = np.uint8(meta['labels'])
            else:
                raise ValueError("SEGMENTATION can only be GT or PCNN")

            # get obj ids, num objs, obj names
            frame_obj_ids = scene_obj_ids[frame_target_indices] if SEGMENTATION == "GT" else np.unique(labels)[1:]
            frame_num_objs = len(frame_obj_ids)
            frame_obj_names = [obj_names[int(idx)] for idx in frame_obj_ids]

            # get extrinsics and intrinsics
            frame_camera = scene_camera[str(frame)]

            camera_extrinsics = np.matrix(np.eye(4))
            # camera_extrinsics[:3, :3] = np.matrix(frame_camera["cam_R_w2c"]).reshape(3, 3).T
            # camera_extrinsics[:3, 3] = (-np.matrix(frame_camera["cam_R_w2c"]).reshape(3, 3).T * np.matrix(frame_camera["cam_t_w2c"]).T) / 1000.0
            camera_extrinsics[:3, :3] = np.matrix(frame_camera["cam_R_w2c"]).reshape(3, 3)
            camera_extrinsics[:3, 3] = np.matrix(frame_camera["cam_t_w2c"]).T / 1000.0

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
            obj_ids_ = [int(v) for v in meta['rois'][:, 1]]
            obj_poses = meta['poses_icp']
            obj_Ts = []
            hypotheses = []
            estimates = []
            refiner_params = []
            n_obj = 0

            # TODO still a bit leaky -- check using fast.ai GPUMemTrace
            gc.collect()
            torch.cuda.empty_cache()

            for obj_id, obj_pose, obj_roi in zip(obj_ids_, obj_poses, meta['rois']):
                objects.append(obj_id)
                # obj_q = obj_pose[:4]
                # obj_pose[:3] = obj_q[1:]
                # obj_pose[3] = obj_q[0]
                #
                # obj_T = np.matrix(np.eye(4))
                # obj_T[:3, :3] = Rotation.from_quat(obj_pose[:4]).as_dcm()
                # obj_T[:3, 3] = obj_pose[4:].reshape(3, 1)
                #
                # obj_confidence = 1.0
                #
                # obj_Ts.append(obj_T)

                obj_mask = labels == obj_id
                mask_ids = np.argwhere(labels == obj_id)
                # obj_roi = [np.min(mask_ids[:, 0]), np.min(mask_ids[:, 1]), np.max(mask_ids[:, 0]), np.max(mask_ids[:, 1])]


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


                vmin, vmax, umin, umax = get_bbox(obj_roi)
                obj_roi = [vmin, umin, vmax, umax]

                # hypotheses.append(Hypothesis("%0.2d" % obj_id, obj_T, obj_roi, obj_mask, None, None, 0, obj_confidence))
                # estimate = [obj_q, obj_pose[4:].reshape(3, 1), obj_confidence]
                # _, _, _, emb, cloud = df.forward(rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id)  # TODO use this to generate hypotheses using DF

                if EST_MODE == "DF":
                    estimates, emb, cloud = df.estimate(rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id,
                                                        Verefine.HYPOTHESES_PER_OBJECT)
                elif REF_MODE == "DF":
                    _, _, _, emb, cloud = df.forward(rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id)
                else:
                    emb, cloud = None, None

                if emb is not None or REF_MODE != "DF":
                    if EST_MODE == "PCS":
                        path = "/home/dominik/projects/hsr-grasping/data/pcs_hypotheses/%s/%0.2d_%0.4d/%s_result.txt" \
                               % ("pcnn" if SEGMENTATION == "PCNN" else "gt", scene, frame, obj_names[obj_id])
                        if not os.path.exists(path):
                            continue  # no hypotheses for this object
                        else:
                            n_obj += 1
                        with open(path, 'r') as file:
                            obj_hypotheses = file.readlines()

                        new_hypotheses = []
                        new_refiner_params = []
                        confidences = []
                        for hi, obj_hypothesis in enumerate(obj_hypotheses):
                            parts = obj_hypothesis.split(" ")
                            parts = [float(v.replace("\n", "")) for v in parts]
                            q = np.array(parts[4:-1] + [parts[3]])
                            t = np.array(parts[:3])
                            c = parts[-1]  # TODO is low good or is high good?

                            obj_T = np.matrix(np.eye(4))
                            obj_T[:3, :3] = Rotation.from_quat(q).as_dcm()
                            obj_T[:3, 3] = t.reshape(3, 1)
                            estimate = [q, t, c]

                            refiner_param = [rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id, estimate,
                                             Verefine.ITERATIONS, emb, cloud]
                            new_hypotheses.append(
                                Hypothesis("%0.2d" % obj_id, obj_T, obj_roi, obj_mask, None, None, hi, c,
                                           refiner_param=refiner_param))
                            # new_refiner_params.append(
                            #     [rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id, estimate,
                            #      Verefine.ITERATIONS, emb, cloud])
                            confidences.append(c)

                        # only select the [HYPOTHESES_PER_OBJECT] best hypotheses
                        best_estimates = np.argsort(confidences)[::-1][:Verefine.HYPOTHESES_PER_OBJECT]
                        hypotheses += [[new_hypotheses[idx] for idx in best_estimates]]
                        refiner_params += [[new_refiner_params[idx] for idx in best_estimates]]
                    else:  # DF
                        n_obj += 1

                        # a) take n estimates
                        new_hypotheses = []
                        new_refiner_params = []
                        for hi, estimate in enumerate(estimates):
                            obj_T = np.matrix(np.eye(4))
                            obj_T[:3, :3] = Rotation.from_quat(estimate[0]).as_dcm()
                            obj_T[:3, 3] = estimate[1].reshape(3, 1)
                            obj_confidence = estimate[2]
                            refiner_param = [rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id, estimate,
                                             Verefine.ITERATIONS, emb, cloud]
                            new_hypotheses.append(Hypothesis("%0.2d" % obj_id, obj_T, obj_roi, obj_mask, None, None, hi,
                                                             obj_confidence, refiner_param=refiner_param))
                            # new_refiner_params.append([rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id, estimate, Verefine.ITERATIONS, emb, cloud])
                            if REF_MODE == "ICP":
                                refiner_params[-1][-2:] = None, None
                        hypotheses += [new_hypotheses]
                        refiner_params += [new_refiner_params]

                    # # b) perturb best estimate
                    # estimate = estimates[0]
                    #
                    # obj_T = np.matrix(np.eye(4))
                    # obj_rotation = np.matrix(Rotation.from_quat(estimate[0]).as_dcm())
                    # # obj_t = estimate[1].reshape(3, 1)
                    # obj_T[:3, :3] = Rotation.from_quat(estimate[0]).as_dcm()
                    # obj_T[:3, 3] = estimate[1].reshape(3, 1)
                    # obj_confidence = estimate[2]
                    #
                    # hypotheses.append(
                    #     Hypothesis("%0.2d" % obj_id, obj_T.copy(), obj_roi, obj_mask, None, None, 0, obj_confidence))
                    # refiner_params.append(
                    #     [rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id, estimate, Verefine.ITERATIONS, emb,
                    #      cloud])
                    #
                    # for axis in ['x', 'y', 'z']:
                    #     for angle in [-5, 5]:
                    #         new_rotation = np.matrix(Rotation.from_euler(axis, angle, degrees=True).as_dcm()) * obj_rotation
                    #
                    #         new_estimate = [Rotation.from_dcm(new_rotation).as_quat(), estimate[1], estimate[2]]
                    #         new_obj_T = np.matrix(np.eye(4))
                    #         new_obj_T[:3, :3] = new_rotation#.as_dcm()
                    #         new_obj_T[:3, 3] = new_estimate[1].reshape(3, 1)
                    #
                    #         estimate = new_estimate.copy()
                    #         obj_T = new_obj_T.copy()
                    #
                    #         hypotheses.append(
                    #             Hypothesis("%0.2d" % obj_id, obj_T, obj_roi, obj_mask, None, None, 0, obj_confidence))
                    #         refiner_params.append(
                    #             [rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id, estimate,
                    #              Verefine.ITERATIONS, emb, cloud])

            # --- refine
            final_hypotheses = []
            st = time.time()

            if MODE != "BASE":
                # init frame
                simulator.initialize_frame(camera_extrinsics)
            if MODE != "BASE" or REF_MODE == "ICP":
                obs = depth.reshape(480, 640, 1)
            #     obs[labels == 0] = 0  # makes results worse (with PCNN seg)
                renderer.set_observation(obs)

            if MODE == "BASE":
                # DF (base)
                refinements = 0
                # refine only max conf
                for obj_hypotheses in hypotheses:
                    hypothesis = obj_hypotheses[0]  # only take best

                    # refinements += refiner_param[-3]
                    q, t, c = ref.refine(*hypothesis.refiner_param)

                    hypothesis.transformation[:3, :3] = Rotation.from_quat(q).as_dcm()
                    hypothesis.transformation[:3, 3] = t.reshape(3, 1)
                    hypothesis.confidence = c
                    final_hypotheses.append(hypothesis)
                # print(refinements)

            elif MODE == "PIR":
                # PIR
                for obj_hypotheses in hypotheses:
                    hypothesis = obj_hypotheses[0]  # only take best
                    phys_hypotheses = pir.refine(hypothesis)

                    final_hypotheses.append(phys_hypotheses[-1])  # pick hypothesis after last refinement step

            elif MODE == "BAB":
                # BAB (with PIR)
                observation = {
                    "depth": depth,
                    "extrinsics": camera_extrinsics,
                    "intrinsics": camera_intrinsics
                }
                refinements = 0
                for obj_hypotheses in hypotheses:
                    # set according to actual number of hypotheses (could be less for PCS if we don't find enough)
                    Verefine.MAX_REFINEMENTS_PER_HYPOTHESIS = Verefine.ITERATIONS * Verefine.REFINEMENTS_PER_ITERATION * len(obj_hypotheses)

                    bab = BudgetAllocationBandit(pir, observation, obj_hypotheses)
                    bab.refine_max()
                    hypothesis, plays, fit = bab.get_best()
                    assert hypothesis is not None
                    final_hypotheses.append(hypothesis)

                    refinements += bab.max_iter - len(obj_hypotheses)  # don't count initial render
                # print(refinements)

            elif MODE == "SV":
                pass  # TODO this should be similar to mitash, just everything at scene level -> no BAB, adapt candidate generation

            elif MODE == "VF":
                observation = {
                    "depth": depth,
                    "extrinsics": camera_extrinsics,
                    "intrinsics": camera_intrinsics
                }
                hypotheses_pool = dict()
                for obj_hypotheses in hypotheses:
                    hypotheses_pool[obj_hypotheses[0].model] = obj_hypotheses

                Verefine.OBSERVATION = observation
                final_hypotheses, final_fit, convergence_hypotheses = Verefine.verefine_solution(hypotheses_pool)

                if Verefine.TRACK_CONVERGENCE:
                    # fill-up to 200 with final hypothesis
                    convergence_hypotheses += [final_hypotheses] * (200 - len(convergence_hypotheses))

                    # write results
                    for convergence_iteration, iteration_hypotheses in enumerate(convergence_hypotheses):
                        iteration_hypotheses = [hs[0] for hs in iteration_hypotheses]
                        for hypothesis in iteration_hypotheses:
                            with open("/home/dominik/projects/hsr-grasping/convergence-vf2/%0.3d_ycbv-test.csv"
                                      % convergence_iteration, 'a') as file:
                                parts = ["%0.2d" % scene, "%i" % frame, "%i" % int(hypothesis.model),
                                         "%0.3f" % hypothesis.confidence,
                                         " ".join(
                                             ["%0.6f" % v for v in np.array(hypothesis.transformation[:3, :3]).reshape(9)]),
                                         " ".join(
                                             ["%0.6f" % (v * 1000) for v in np.array(hypothesis.transformation[:3, 3])]),
                                         "%0.3f" % 1.0]
                                file.write(",".join(parts) + "\n")

                final_hypotheses = [hs[0] for hs in final_hypotheses]

            durations.append(time.time() - st)

            # write results
            for hypothesis in final_hypotheses:
                with open("/home/dominik/projects/hsr-grasping/vf2_ycbv-test.csv", 'a') as file:
                    parts = ["%0.2d" % scene, "%i" % frame, "%i" % int(hypothesis.model), "%0.3f" % hypothesis.confidence,
                             " ".join(["%0.6f" % v for v in np.array(hypothesis.transformation[:3, :3]).reshape(9)]),
                             " ".join(["%0.6f" % (v * 1000) for v in np.array(hypothesis.transformation[:3, 3])]),
                             "%0.3f" % durations[-1]]
                    file.write(",".join(parts) + "\n")

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
