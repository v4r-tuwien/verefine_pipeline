import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np
import json
import scipy.io as scio
from scipy.spatial.transform.rotation import Rotation
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
from src.verefine.verefine import Hypothesis, PhysIR, BudgetAllocationBandit
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
PATH_APC = "/home/dominik/experiments/PhysimGlobalPose/src/dataset_baseline/"#_fcnn-pcs-lcp/"
PATH_APC_old = "/home/dominik/experiments/PhysimGlobalPose/src/dataset/"

POOL = "clusterPose"  # "allPose" for Super4PCS(?) ordered by LCP, "clusterPose" for cluster hypotheses (exactly 25), "super4pcs" for Super4PCS (best LCP)
MODE = "BAB"  # "BASE", "PIR", "BAB", "SV", "VF"
EST_MODE = "PCS"
REF_MODE = "ICP"  # TODO ICP not the same

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

# -----------------

if __name__ == "__main__":

    dataset = ExApcDataset(base_path=PATH_APC)
    ref = None
    durations = []
    errors_translation, errors_rotation = [], []

    # if MODE != "BASE" or REF_MODE == "ICP":
    renderer = Renderer(dataset)
    Verefine.RENDERER = renderer
        # renderer.create_egl_context()

    if MODE != "BASE":
        simulator = Simulator(dataset, instances_per_object=25)
        Verefine.SIMULATOR = simulator
        pir = None

    with open(PATH_APC + "/obj_config_apc.yml", 'r') as file:
        obj_config = yaml.load(file)
    symInfos = {}
    for k, v in obj_config['objects'].items():
        if not "object_" in k:
            continue
        symInfos[v['classId']] = v['symmetry']

    # loop over scenes...
    objects = []
    scenes = list(range(1, 31))
    for scene in scenes:
        print("scene %i..." % scene)

        # gt info
        with open(PATH_APC + "scene-%0.4d/gt_info.yml" % scene, 'r') as file:
            gt_info = yaml.load(file)

        # load observation
        rgb = np.array(PIL.Image.open(PATH_APC + "scene-%0.4d/frame-000000.color.png" % scene))
        depth = np.array(PIL.Image.open(PATH_APC + "scene-%0.4d/frame-000000.depth.png" % scene), dtype=np.uint16)
        depth = (depth << 13 | depth >> 3)/10  # [mm]
        scene_depth = np.array(PIL.Image.open(PATH_APC + "scene-%0.4d/debug_search/scene.png" % scene), dtype=np.uint16)  # table removed (as in Mitash)
        scene_depth = (scene_depth << 13 | scene_depth >> 3) / 10  # [mm]
        depth = scene_depth

        # camera data
        camera_intrinsics = np.array(gt_info['camera']['camera_intrinsics'])
        camera_extrinsics = np.matrix(np.eye(4))
        # note: translation relative to table (rotation of table is always I)
        # camera_extrinsics[:3, 3] = np.matrix(gt_info['camera']['camera_pose'][:3]).T
        camera_extrinsics[:3, 3] = (np.matrix(gt_info['camera']['camera_pose'][:3])
                                    - np.matrix(gt_info['rest_surface']['surface_pose'][:3])).T  # [m]
        camera_q = gt_info['camera']['camera_pose'][3:]  # wxyz
        camera_q = camera_q[1:] + [camera_q[0]]  # xyzw
        camera_extrinsics[:3, :3] = Rotation.from_quat(camera_q).as_dcm()


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
            # "rgb": rgb,  # TODO debug
            "depth": depth,
            "normals": estimate_normals(depth/1000),
            "extrinsics": camera_extrinsics,
            "intrinsics": camera_intrinsics
        }

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
        obj_ids = []
        obj_gt_Ts = {}
        hypotheses = []
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
            obj_gt_T[:3, 3] = (np.matrix(obj_info['pose'][:3])
                               - np.matrix(gt_info['rest_surface']['surface_pose'][:3])).T
            obj_q = obj_info['pose'][3:]  # wxyz
            obj_q = obj_q[1:] + [obj_q[0]]  # xyzw
            obj_gt_T[:3, :3] = Rotation.from_quat(obj_q).as_dcm()

            # # to old format
            # with open(PATH_APC_old + "scene-%0.4d/gt_pose_%s.txt" % (scene, obj_name), 'w') as file:
            #     file.write(str(obj_gt_T[:3, :]).replace("[", "").replace("]", "").replace("\n", "") + "\n")
            # continue

            # obj_gt_T[:3, 3] = np.matrix(obj_info['pose'][:3]).T - obj_gt_T[:3, :3]*np.matrix(gt_info['rest_surface']['surface_pose'][:3]).T
            world_to_cam = camera_extrinsics.copy()
            world_to_cam[:3, :3] = world_to_cam[:3, :3].T
            world_to_cam[:3, 3] = -world_to_cam[:3, :3] * world_to_cam[:3, 3]
            # world_to_cam = camera_extrinsics.I
            obj_gt_T = world_to_cam * obj_gt_T  # to camera coordinates TODO correct?

            obj_ids.append(obj_id)
            obj_gt_Ts[obj_id] = obj_gt_T

            # -- hyp
            path = PATH_APC + "scene-%0.4d/debug_super4PCS/%s_%s.txt" % (scene, POOL, obj_name)
            if not os.path.exists(path):
                # TODO count as FN
                continue  # no hypotheses for this object
            else:
                n_obj += 1
            with open(path, 'r') as file:
                obj_hypotheses = file.readlines()

            if POOL != "super4pcs":
                with open(PATH_APC + "scene-%0.4d/debug_super4PCS/%s_%s.txt"
                          % (scene, POOL.replace("Pose", "Score"), obj_name), 'r') as file:
                    obj_hypotheses_scores = file.readlines()
                    obj_hypotheses_scores = [float(v.replace("\n", "")) for v in obj_hypotheses_scores]
                score_order = np.argsort(obj_hypotheses_scores)[::-1]
                obj_hypotheses_scores = [obj_hypotheses_scores[i] for i in score_order]
                obj_hypotheses = [obj_hypotheses[i] for i in score_order]
            else:
                obj_hypotheses = [obj_hypotheses[0]]  # TODO 0 is just super4pcs, 1 is with Mitash' TrICP
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

            # scene_depth_down = None
            # downsample and mask observed depth
            # depth_pcd = ref.depth_to_cloud(scene_depth, camera_intrinsics, obj_mask)
            # import open3d as o3d
            # pcd, pcd_down = o3d.PointCloud(), o3d.PointCloud()
            # pcd.points = o3d.Vector3dVector(depth_pcd / 1000)
            # pcd_down = o3d.voxel_down_sample(pcd, voxel_size=0.005)
            # scene_depth_down = np.array(pcd_down.points)
            # print(scene_depth_down.shape)

            # or: load downsampled segment
            with open(
                "/home/dominik/experiments/PhysimGlobalPose/src/dataset/scene-%0.4d/debug_super4PCS/pclSegment_%s.ply"
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
            import open3d as o3d
            pcd = o3d.PointCloud()
            pcd.points = o3d.Vector3dVector(scene_depth_down)
            pcd, ind = o3d.statistical_outlier_removal(pcd, 16, 1.0)
            # pcd, ind = o3d.radius_outlier_removal(pcd, 4, 0.01)
            scene_depth_down = np.array(pcd.points)
            # print(scene_depth_down.shape)


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
                refiner_param = [rgb, obj_depth, camera_intrinsics, None, obj_mask, obj_id, estimate,
                                 Verefine.ITERATIONS, scene_depth_down, None]
                new_hypotheses.append(
                    Hypothesis("%0.2d" % obj_id, obj_T, obj_roi, obj_mask, None, None, hi, c,
                               refiner_param=refiner_param))
                confidences.append(c)

            # only select the [HYPOTHESES_PER_OBJECT] best hypotheses
            best_estimates = np.argsort(confidences)[::-1]#[:Verefine.HYPOTHESES_PER_OBJECT]  # TODO same number as above!
            hypotheses += [[new_hypotheses[idx] for idx in best_estimates]]

            # --- vis
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
                if REF_MODE == "ICP":
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
            refinements = 0
            for obj_hypotheses in hypotheses:

                # obj_depth = depth.copy()
                # obj_mask = obj_hypotheses[0].mask
                # obj_depth[np.logical_not(obj_mask)] = 0
                # observation = {
                #     "depth": obj_depth,
                #     "extrinsics": camera_extrinsics,
                #     "intrinsics": camera_intrinsics
                # }

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
            Verefine.MAX_REFINEMENTS_PER_HYPOTHESIS = Verefine.ITERATIONS * Verefine.REFINEMENTS_PER_ITERATION * len(
                obj_hypotheses)
            pass  # TODO this should be similar to mitash, just everything at scene level -> no BAB, adapt candidate generation

        elif MODE == "VF":
            hypotheses_pool = dict()
            for obj_hypotheses in hypotheses:
                hypotheses_pool[obj_hypotheses[0].model] = obj_hypotheses

            Verefine.MAX_REFINEMENTS_PER_HYPOTHESIS = Verefine.ITERATIONS * Verefine.REFINEMENTS_PER_ITERATION * len(
                obj_hypotheses)
            Verefine.OBSERVATION = observation
            final_hypotheses, final_fit, convergence_hypotheses = Verefine.verefine_solution(hypotheses_pool)

            # if Verefine.TRACK_CONVERGENCE:
                # # fill-up to 200 with final hypothesis
                # convergence_hypotheses += [final_hypotheses] * (200 - len(convergence_hypotheses))

                # # write results
                # for convergence_iteration, iteration_hypotheses in convergence_hypotheses.items():
                #     iteration_hypotheses = [hs[0] for hs in iteration_hypotheses]
                #     for hypothesis in iteration_hypotheses:
                #         with open("/home/dominik/projects/hsr-grasping/convergence-vf5-c1/%0.3d_ycbv-test.csv"
                #                   % convergence_iteration, 'a') as file:
                #             parts = ["%0.2d" % scene, "%i" % 1, "%i" % int(hypothesis.model),
                #                      "%0.3f" % hypothesis.confidence,
                #                      " ".join(
                #                          ["%0.6f" % v for v in np.array(hypothesis.transformation[:3, :3]).reshape(9)]),
                #                      " ".join(
                #                          ["%0.6f" % (v * 1000) for v in np.array(hypothesis.transformation[:3, 3])]),
                #                      "%0.3f" % 1.0]
                #             file.write(",".join(parts) + "\n")

            final_hypotheses = [hs[0] for hs in final_hypotheses]

        durations.append(time.time() - st)

        # --- errors
        errs_t, errs_r = [], []
        for hypothesis in final_hypotheses:
            obj_gt_T = obj_gt_Ts[int(hypothesis.model)]
            err_t = np.linalg.norm(hypothesis.transformation[:3, 3] - obj_gt_T[:3, 3]) * 1000  # [mm]

            symInfo = symInfos[int(hypothesis.model)]
            rotdiff = hypothesis.transformation[:3, :3].I * obj_gt_T[:3, :3]
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
        print("   mean err t = %0.1f" % np.mean(errs_t))
        print("   mean err r = %0.1f" % np.mean(errs_r))

        # print("   refinement iterations = %i" % ref.ref_count)  # TODO
        #
        # # --- vis
        vis = np.dstack((depth, depth, depth))/1000#
        vis = rgb.copy()
        rgb_ren = []
        for hypothesis in final_hypotheses:
            rendered = hypothesis.render(observation, 'color')
            rgb_ren.append(rendered[0])
            # vis[rendered[0] != 0] = vis[rendered[0] != 0] * 0.3 + rendered[0][rendered[0] != 0]/255 * 0.7
            vis[rendered[0] != 0] = vis[rendered[0] != 0] * 0.3 + rendered[0][rendered[0] != 0] * 0.7
        vis2 = rgb.copy()
        for obj_hypotheses in hypotheses:
            rendered = obj_hypotheses[0].render(observation, 'color')
            vis2[rendered[0] != 0] = vis2[rendered[0] != 0] * 0.3 + rendered[0][rendered[0] != 0] * 0.7

        def debug_draw():
            plt.subplot(1, 2, 2)
            plt.imshow(vis)
            plt.subplot(1, 2, 1)
            plt.imshow(vis2)
        drawnow(debug_draw)
        plt.pause(0.05)
        # plt.pause(5.0)

        print("   ---")
        print("   ~ icp/call=%0.1fms" % (np.mean(TrimmedIcp.icp_durations) * 1000))
        if MODE not in ["BASE", "PIR"]:
            print("   ~ rendering/call=%0.1fms" % (np.mean(np.sum(renderer.runtimes, axis=1)) * 1000))
            print("   ~ cost/call=%0.1fms" % (np.mean(Verefine.cost_durations) * 1000))
        print("   ---")
        print("   ~ icp/frame=%ims" % (np.sum(TrimmedIcp.icp_durations) * 1000))
        if MODE not in ["BASE", "PIR"]:
            print("   ~ rendering/frame=%ims" % (np.sum(np.sum(renderer.runtimes, axis=1)) * 1000))
            print("   ~ cost/frame=%ims" % (np.sum(Verefine.cost_durations) * 1000))

        TrimmedIcp.icp_durations.clear()
        renderer.runtimes.clear()
        Verefine.cost_durations.clear()

    print("\n\n")
    print("err t [cm] = %0.1f" % (np.mean(errors_translation)/10))
    print("err r [deg] = %0.1f" % np.mean(errors_rotation))
    print("------")
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
