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

from src.maskrcnn.maskrcnn import MaskRcnnDetector
from src.util.fast_renderer import Renderer
from src.util.dataset import YcbvDataset
from src.densefusion.densefusion import DenseFusion
from src.verefine.simulator import Simulator
from src.verefine.verefine import Hypothesis, PhysIR, BudgetAllocationBandit
import src.verefine.verefine as Verefine
from src.icp.icp import TrimmedIcp
from src.verefine.plane_segmentation import PlaneDetector


# settings
PATH_BOP19 = "/mnt/Data/datasets/BOP19/"
# PATH_LM = "/mnt/Data/datasets/SIXD/LM_LM-O/"
PATH_YCBV_ROOT = '/mnt/Data/datasets/YCB Video/YCB_Video_Dataset'

TEST_TARGETS = "test_targets_bop19"

MODE = "BAB"  # "{BASE, PIR, BAB, MITASH} -> single, {BEST, ALLON, EVEN, BAB} -> multi, {VFlist, VFtree} -> verefine
EST_MODE = "DF"  # "GT", "DF", "PCS"
REF_MODE = "DF"  # "DF", "ICP"
SEGMENTATION = "PCNN"  # GT, PCNN, MRCNN
EXTRINSICS = "PLANE"  # GT, PLANE
TAU = 20
TAU_VIS = 10  # [mm]

SCENES = []

PLOT = False

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

    Verefine.HYPOTHESES_PER_OBJECT = 5
    Verefine.ITERATIONS = 2
    Verefine.SIM_STEPS = 3
    Verefine.C = np.sqrt(2)
    Verefine.fit_fn = Verefine.fit_multi

    dataset = YcbvDataset(base_path=PATH_YCBV_ROOT)
    if SEGMENTATION == "MRCNN":
        maskrcnn = MaskRcnnDetector()
    # SCENES = dataset.objlist[1:]
    #
    # if len(sys.argv) > 1 and len(sys.argv) >= 3:
    #     # refine mode, offset mode and offset size (0 is script path)
    #     MODE = sys.argv[1]
    #     MODE_OFF = sys.argv[2]
    #     # OFFSETS = [int(sys.argv[3])]
    #     SCENES = dataset.objlist[1:] if len(sys.argv) == 3 else [int(sys.argv[3])]
    #     PLOT = False
    # print("mode: %s -- scenes: %s" % (MODE, SCENES))

    df = None
    durations = []

    # if MODE != "BASE" or REF_MODE == "ICP":
    renderer = Renderer(dataset, recompute_normals=False)#(MODE!="BASE"))
    Verefine.RENDERER = renderer
        # renderer.create_egl_context()

    if MODE != "BASE":
        simulator = Simulator(dataset, instances_per_object=Verefine.HYPOTHESES_PER_OBJECT)
        Verefine.SIMULATOR = simulator
        pir = None


    # get all keyframes
    with open(PATH_YCBV_ROOT + "/image_sets/keyframe.txt", 'r') as file:
        keyframes = file.readlines()
    keyframes = [keyframe.replace("\n", "") for keyframe in keyframes]

    # get all test targets
    with open(PATH_BOP19 + "ycbv/%s.json" % TEST_TARGETS, 'r') as file:
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
    scenes = sorted(np.unique(scene_ids))
    for scene in scenes:
        # if scene < 56:
        #     continue
        print("scene %i..." % scene)

        scene_target_indices = np.argwhere(scene_ids == scene)

        # frames and objects in scene
        scene_im_ids = im_ids[scene_target_indices]
        scene_obj_ids = obj_ids[scene_target_indices]

        # camera infos
        with open(PATH_BOP19 + "ycbv/test/%0.6d/scene_camera.json" % scene, 'r') as file:
            scene_camera = json.load(file)

        # scene gt
        with open(PATH_BOP19 + "ycbv/test/%0.6d/scene_gt.json" % scene, 'r') as file:
            scene_gt = json.load(file)
        with open(PATH_BOP19 + "ycbv/test/%0.6d/scene_gt_info.json" % scene, 'r') as file:
            scene_gt_info = json.load(file)

        # loop over frames in scene...
        frames = sorted(np.unique(scene_im_ids))
        for fi, frame in enumerate(frames):
            # if frame < 1049:
            #     continue
            print("   frame %i (%i/%i)..." % (frame, fi+1, len(frames)))

            # objects in frame
            frame_target_indices = np.argwhere(scene_im_ids == frame)[:, 0]

            # # meta data (PCNN segmentation and poses)
            # keyframe = keyframes.index("%0.4d/%0.6d" % (scene, frame))
            # meta = scio.loadmat(PATH_LM + "../YCB_Video_toolbox/results_PoseCNN_RSS2018/%0.6d.mat" % keyframe)

            # load observation
            rgb = np.array(PIL.Image.open(PATH_BOP19 + "ycbv/test/%0.6d/rgb/%0.6d.png" % (scene, frame)))
            depth = np.float32(np.array(PIL.Image.open(PATH_BOP19 + "ycbv/test/%0.6d/depth/%0.6d.png" % (scene, frame)))) / 10

            if SEGMENTATION == "GT":
                raise ValueError("GT not ready -- need to merge all labels from all files")
                labels = np.array(PIL.Image.open(PATH_BOP19 + "ycbv/test/%0.6d/mask/%0.6d_000000.png" % (scene, frame)))
                rois = []  # TODO
            elif SEGMENTATION == "PCNN":
                # if not os.path.exists(PATH_YCBV_ROOT + "data/%0.2d_label/%0.4d_label.png" % (scene, frame)):
                #     print("no segmentation for %i-%i - skipping frame" % (scene, frame))
                #     continue
                # labels = np.array(PIL.Image.open(PATH_LM_ROOT + "segnet_results/%0.2d_label/%0.4d_label.png" % (scene, frame)))
                # if (labels > 0).sum() == 0:
                #     print("no segmentation for %i-%i - skipping frame" % (scene, frame))
                #     continue
                toolbox_idx = keyframes.index("%0.4d/%0.6d" % (scene, frame))
                meta = scio.loadmat(PATH_YCBV_ROOT + "/../YCB_Video_toolbox/results_PoseCNN_RSS2018/%0.6d.mat" % toolbox_idx)
                labels = meta['labels']
                rois = meta['rois']
            elif SEGMENTATION == "MRCNN":
                obj_ids_, rois, masks, scores = maskrcnn.detect(rgb)

                to_delete = []
                test = [masks[:, :, i] for i in range(len(scores))]
                for mi, (mask, score) in enumerate(zip(test, scores)):
                    for other_mi, (other_mask, other_score) in enumerate(zip(test, scores)):
                        if mi == other_mi:
                            continue
                        if np.logical_and(mask>0, other_mask>0).sum() > (mask>0).sum() * 0.5:
                            to_delete.append(mi if score < other_score else other_mi)

                masks = np.dstack([t for mi, t in enumerate(test) if mi not in to_delete])
                obj_ids_ = [o for mi, o in enumerate(obj_ids_) if mi not in to_delete]
                rois = [roi for mi, roi in enumerate(rois) if mi not in to_delete]

                rois = [[0, obj_id] + [roi[1], roi[0], roi[3], roi[2]] for obj_id, roi in zip(obj_ids_, rois)]
                labels = masks.sum(axis=2)
            else:
                raise ValueError("SEGMENTATION can only be GT or PCNN")

            # get obj ids, num objs, obj names
            frame_obj_ids = scene_obj_ids[frame_target_indices] if SEGMENTATION == "GT" else [roi[1] for roi in rois]
            frame_num_objs = 1  # len(frame_obj_ids)
            frame_obj_names = [obj_names[int(idx)] for idx in frame_obj_ids]

            # get extrinsics and intrinsics
            frame_camera = scene_camera[str(frame)] if "ral" not in TEST_TARGETS else scene_camera[list(scene_camera.keys())[0]]

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
            # obj_ids_ = [scene]  #[int(v) for v in meta['rois'][:, 1]]
            # pose_info = scene_gt['%i' % frame][0]  # TODO load GT separately when not using BOP19
            pose = np.matrix(np.eye(4))
            # pose[:3, :3] = np.array(pose_info['cam_R_m2c']).reshape(3, 3)
            # pose[:3, 3] = np.array(pose_info['cam_t_m2c']).reshape(3, 1)/1000
            obj_poses = [pose]

            if EXTRINSICS == "PLANE":
                plane_detector = PlaneDetector(640, 480, camera_intrinsics, down_scale=4)

                # camera_extrinsics = np.matrix(np.eye(4))
                # camera_extrinsics[:3, :3] = np.matrix(frame_camera["cam_R_w2c"]).reshape(3, 3)
                # camera_extrinsics[:3, 3] = np.matrix(frame_camera["cam_t_w2c"]).T / 1000.0
                try:
                    camera_extrinsics = plane_detector.detect(depth, labels > 0)
                except ValueError as ex:
                    print(ex)
                    print("skipping frame...")
                    continue
            elif EXTRINSICS == "GT":
                meta = scio.loadmat(
                    PATH_YCBV_ROOT + "/data/%0.4d/%0.6d-meta.mat" % (scene, frame))
                camera_extrinsics = np.matrix(np.eye(4))
                camera_extrinsics[:3, :] = meta['rotation_translation_matrix']
                # camera_extrinsics = pose.copy()
                # camera_extrinsics[:3, 3] = camera_extrinsics[:3, 3] + camera_extrinsics[:3, :3]*np.matrix([0.0, 0.0, dataset.obj_bot[scene-1]]).T
            else:
                raise ValueError("EXTRINSICS needs to be GT (from model) or PLANE (segmentation).")

            # TODO debug vis for extrinsics
            # vis = depth.copy()
            # renderer.set_observation(np.zeros_like(rgb), np.zeros_like(depth))
            # plane = renderer.render([0], [camera_extrinsics], camera_extrinsics, camera_intrinsics, mode='depth')[1]*1000
            # plane = np.abs(plane - depth) < 5
            # vis[plane>0] = 1
            # vis[np.logical_and(plane==0, depth>0)] = -1
            # plt.imshow(vis)
            # plt.show()
            # continue

            # TODO still a bit leaky -- check using fast.ai GPUMemTrace
            gc.collect()
            torch.cuda.empty_cache()

            scene_depth = depth.copy()
            scene_depth[labels == 0] = 0
            ref_depth = scene_depth.copy()  # TODO always? or only for experiments?


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

            hypotheses = []
            offsets = []
            refiner_params = []

            # TODO still a bit leaky -- check using fast.ai GPUMemTrace
            gc.collect()
            torch.cuda.empty_cache()

            for oi, obj_roi in enumerate(rois):
                obj_id = int(obj_roi[1])
                class_id = obj_id

                if SEGMENTATION == "MRCNN":
                    obj_mask = masks[:, :, oi]
                else:
                    obj_mask = labels == obj_id

                vmin, vmax, umin, umax = get_bbox(obj_roi)
                obj_roi = [vmin, umin, vmax, umax]

                if EST_MODE == "DF":
                    estimates, emb, cloud = df.estimate(rgb, ref_depth, camera_intrinsics, obj_roi, obj_mask, class_id,
                                                        Verefine.HYPOTHESES_PER_OBJECT)
                elif REF_MODE == "DF":
                    try:
                        _, _, _, emb, cloud = df.forward(rgb, ref_depth, camera_intrinsics, obj_roi, obj_mask, class_id)
                    except ZeroDivisionError:
                        print("empty depth")
                        continue
                else:
                    emb, cloud = None, None

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
                            refiner_param = [rgb, ref_depth, camera_intrinsics, obj_roi, obj_mask, obj_id, estimate,
                                             Verefine.ITERATIONS, emb, cloud, None]
                            new_hypotheses.append(Hypothesis("%0.2d" % obj_id, obj_T, obj_roi, obj_mask, None, None, hi,
                                                             obj_confidence, refiner_param=refiner_param))
                            # new_refiner_params.append([rgb, depth, camera_intrinsics, obj_roi, obj_mask, obj_id, estimate, Verefine.ITERATIONS, emb, cloud])

                            if REF_MODE == "ICP":
                                refiner_params[-1][-3:-1] = None, None
                        hypotheses += [new_hypotheses]
                        refiner_params += [new_refiner_params]


            # --- dependency graph
            if MODE in ["VFlist", "VFtree"]:
                def point_inside(rect, pt):
                    return rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]

                if PLOT:
                    plt.imshow(labels)  # TODO value by id

                # if SEGMENTATION == "MRCNN":
                # a) as is
                obj_ids_ = [roi[1] for roi in rois]
                masks = [hs[0].mask for hs in hypotheses]

                label_count = {}
                label_strs = []
                for o in obj_ids_:
                    if o not in label_count:
                        label_count[o] = 0
                    label_strs.append("%i (%i)" % (o, label_count[o]))
                    label_count[o] += 1
                #
                # if SEGMENTATION == "PCNN":
                #     # # b) multiple instances
                #     from skimage import measure
                #
                #     connected_labels = measure.label(labels)
                #     connected_ids = sorted(np.unique(connected_labels))[1:]
                #     new_labels = np.zeros_like(labels)
                #     label_count = {}
                #     obj_ids_ = []
                #     masks = []
                #     label_strs = []
                #     for connected_id in connected_ids:
                #         connected_mask = connected_labels == connected_id
                #         if (connected_mask > 0).sum() < 1000:
                #             continue
                #         masks.append(connected_mask)
                #         label_id = int(np.unique(labels[connected_mask])[0])
                #         obj_ids_.append(label_id)
                #         if label_id in label_count:
                #             label_count[label_id] += 1
                #             label_strs.append("%i (%i)" % (label_id, label_count[label_id]))
                #             label_id += 30 * label_count[label_id]
                #         else:
                #             label_count[label_id] = 0
                #             label_strs.append("%i (%i)" % (label_id, label_count[label_id]))
                #         new_labels[connected_mask] = label_id
                #         # plt.imshow(new_labels)

                rois = [list(np.min(np.argwhere(mask > 0), axis=0)) + list(np.max(np.argwhere(mask > 0), axis=0)) for
                        mask in masks]
                overlapping = []
                occluded = []
                supporting = []
                centers = []
                for roi, mask in zip(rois, masks):
                    centers.append(np.mean(np.argwhere(mask > 0), axis=0))

                    umap = np.array([[j for _ in range(640)] for j in range(480)])
                    vmap = np.array([[i for i in range(640)] for _ in range(480)])


                    # TODO to world coords
                    def to_world(px_mask):
                        fx, fy, cx, cy = camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], \
                                         camera_intrinsics[1, 2]
                        valid_mask = np.logical_and(px_mask > 0, depth > 0)
                        D_masked = depth[valid_mask > 0] / 1000  # in meters
                        X_masked = (vmap[valid_mask > 0] - cx) * D_masked / fx
                        Y_masked = (umap[valid_mask > 0] - cy) * D_masked / fy
                        return np.dot(np.dstack([X_masked, Y_masked, D_masked]),
                                      camera_extrinsics[:3, :3]) + camera_extrinsics.I[:3, 3].T


                    center_z = to_world(mask)[:, 2].mean()

                    overlap_obj = []
                    occluded_obj = []
                    supporting_obj = []
                    for other_roi, other_mask in zip(rois, masks):
                        check = point_inside(other_roi, [roi[0], roi[1]])  # tl
                        check |= point_inside(other_roi, [roi[0], roi[3]])  # tr
                        check |= point_inside(other_roi, [roi[2], roi[3]])  # br
                        check |= point_inside(other_roi, [roi[2], roi[1]])  # bl

                        if PLOT:
                            plt.plot([roi[1], roi[3]], [roi[0], roi[0]], 'r-')  # t
                            plt.plot([roi[3], roi[3]], [roi[0], roi[2]], 'r-')  # r
                            plt.plot([roi[3], roi[1]], [roi[2], roi[2]], 'r-')  # b
                            plt.plot([roi[1], roi[1]], [roi[2], roi[0]], 'r-')  # l

                        overlap_obj.append(check)
                        occluded_obj.append(check and np.mean(mask) > np.mean(other_mask))
                        center = np.mean(np.argwhere(mask > 0), axis=0)
                        other_center = np.mean(np.argwhere(other_mask > 0), axis=0)

                        other_center_z = to_world(other_mask)[:, 2].mean()
                        supporting_obj.append(bool(center_z < other_center_z))
                    overlapping.append(overlap_obj)
                    occluded.append(occluded_obj)
                    supporting.append(supporting_obj)
                overlapping = np.logical_or(np.array(overlapping), np.array(overlapping).T)

                # TODO cluster
                # # bandwidth reduction -> permutation s.t. distance on nonzero entries from the center diagonal is minimized
                # adjacency = np.uint8(overlapping)
                # from scipy.sparse import csgraph
                # r = csgraph.reverse_cuthill_mckee(csgraph.csgraph_from_dense(adjacency), True)
                #
                # # via http://raphael.candelier.fr/?blog=Adj2cluster
                # # and http://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html
                # # -> results in blocks in the adjacency matrix that correspond with the clusters
                # # -> iteratively extend the block while the candidate region contains nonzero elements (i.e. is connected)
                # clusters = [[r[0]]]
                # for i in range(1, len(r)):
                #     if np.any(adjacency[clusters[-1], r[i]]):  # ri connected to current cluster? -> add ri to cluster
                #         clusters[-1].append(r[i])
                #     else:  # otherwise: start a new cluster with ri
                #         clusters.append([r[i]])
                #
                # # add clustered objects to hypotheses clusters
                # adjacencies = []
                # for cluster in clusters:
                #     cluster_pool = {}
                #     cluster_adjacency = {}
                #     for ci in cluster:
                #         # TODO write this python-y
                #         cluster_adjacency[obj_str] = []
                #         for ci_ in cluster:
                #             if ci_ == ci:
                #                 continue
                #             if adjacency[ci, ci_] == 1:
                #                 cluster_adjacency[obj_str].append(obj_ids[ci_])
                #     adjacencies.append(cluster_adjacency)

                test = np.array(np.logical_or(supporting, np.eye(len(supporting))))
                order = []
                for i in range(len(supporting)):
                    # print(test)
                    best = np.argmax(np.sum(test, axis=1))
                    if np.sum(test[best, :]) == 0:
                        print("TODO here consider occlusion")
                    order.append(best)
                    test[best, :] = False
                tree = [label_strs[o] for o in order]
                trees = [order]  # TODO per cluster

                if PLOT:
                    pos = dict()
                    for o in order:
                        pos[label_strs[o]] = [centers[o][1], centers[o][0]]
                    import networkx as nx

                    G = nx.DiGraph()
                    for li, label_str in enumerate(tree):
                        G.add_node(label_str)
                        if li < len(tree) - 1:
                            G.add_edge(label_str, tree[li + 1])
                    nx.draw(G, pos=pos, with_labels=True)
                    plt.show()
                    # continue

            # --- refine
            final_hypotheses = []
            st = time.time()

            if MODE != "BASE":
                # init frame
                simulator.initialize_frame(camera_extrinsics)
            # if MODE != "BASE" or REF_MODE == "ICP":
            renderer.set_observation(scene_depth.reshape(480, 640, 1), scene_normals.reshape(480, 640, 3))

            observation = {
                # "rgb": rgb,  # TODO debug
                "depth": scene_depth,
                "normals": scene_normals,
                "extrinsics": camera_extrinsics,
                "intrinsics": camera_intrinsics,
                "mask_others": np.zeros_like(scene_depth)
            }

            # TODO debug -- plot all hypotheses
            # if PLOT:
            #     def dbg():
            #         # for u in range(len(hypotheses)):
            #         #     for v in range(len(hypotheses[0])):
            #         #         plt.subplot(len(hypotheses), len(hypotheses[0]), u * 5 + v + 1)
            #         #         plt.imshow(hypotheses[u][v].render(observation, 'color')[0])
            #         for u in range(min(len(hypotheses), 6)):
            #             all = np.zeros_like(rgb)
            #             for v in range(len(hypotheses[0])):
            #                 cur = hypotheses[u][v].render(observation, 'color')[0]
            #                 all[all == 0] = cur[all == 0]
            #             plt.subplot(3, 2, u + 1)
            #             plt.imshow(all)
            #     drawnow(dbg)
            #     plt.pause(2.0)

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

                # observation = {
                #     "color": rgb,
                #     "depth": scene_depth,
                #     "normals": scene_normals,
                #     "extrinsics": camera_extrinsics,
                #     "intrinsics": camera_intrinsics,
                #     # "label": obj_hypotheses[0].mask  # TODO move into loop to set this
                # }

                # BAB (with PIR)
                refinements = 0
                for obj_hypotheses in hypotheses:

                    # set according to actual number of hypotheses (could be less for PCS if we don't find enough)
                    # Verefine.MAX_REFINEMENTS_PER_HYPOTHESIS = Verefine.ITERATIONS * Verefine.REFINEMENTS_PER_ITERATION * len(obj_hypotheses)

                    if Verefine.fit_fn == Verefine.fit_multi:
                        renderer.set_observation(scene_depth.reshape(480, 640, 1), scene_normals.reshape(480, 640, 3),
                                                 obj_hypotheses[0].mask.reshape(480, 640, 1)==0)

                    bab = BudgetAllocationBandit(pir, observation, obj_hypotheses)
                    bab.refine_max()

                    # # TODO debug plot
                    # if PLOT:
                    #     for oi, hs in enumerate(bab.pool):
                    #         for hi, h in enumerate(hs):
                    #             if h is None:
                    #                 continue
                    #             plt.subplot(len(bab.pool), len(hs), oi * len(hs) + hi + 1)
                    #             # ren_d = h.render(bab.observation, 'depth')[1] * 1000
                    #             # vis = np.abs(ren_d - bab.observation['depth'])
                    #             # vis[ren_d == 0] = 0
                    #             ren_c = h.render(bab.observation, 'color')[0]
                    #             vis = ren_c/255*0.7 + rgb/255*0.3
                    #             plt.imshow(vis[h.roi[0]-50:h.roi[2]+50, h.roi[1]-50:h.roi[3]+50])
                    #             plt.title("%0.2f" % bab.fits[oi, hi])
                    #     plt.show()

                    hypothesis, plays, fit = bab.get_best()
                    assert hypothesis is not None
                    final_hypotheses.append(hypothesis)

                    refinements += bab.max_iter - len(obj_hypotheses)  # don't count initial render
                # print(refinements)

            elif MODE == "EVEN":  # spend refinement iterations evenly, select best based on score
                for obj_hypotheses in hypotheses:
                    even_obj_hypotheses = []
                    even_obj_scores = []
                    for hypothesis in obj_hypotheses:
                        phys_hypothesis = pir.refine(hypothesis)[-1]  # pick hypothesis after last refinement step
                        even_obj_hypotheses.append(phys_hypothesis)
                        even_obj_scores.append(phys_hypothesis.fit(observation))
                    final_hypotheses.append(even_obj_hypotheses[int(np.argmax(even_obj_scores))])
            elif MODE in ["ALLON", "BEST"]:
                for obj_hypotheses in hypotheses:
                    allon_obj_scores = []
                    for hypothesis in obj_hypotheses:
                        allon_obj_scores.append(hypothesis.fit(observation))
                    best_hypothesis = obj_hypotheses[int(np.argmax(allon_obj_scores))]
                    if MODE == "ALLON":  # spend full budget
                        pir_hypothesis = pir.refine(best_hypothesis, override_iterations=Verefine.ITERATIONS *
                                                                                          len(obj_hypotheses))[-1]
                    else:  # one full refinement (like BASE)
                        pir_hypothesis = pir.refine(best_hypothesis)[-1]
                    final_hypotheses.append(pir_hypothesis)

            elif MODE == "VFlist":
                # BAB (with PIR)
                refinements = 0

                obj_depths = [hs[0].mask for hs in hypotheses]

                for tree in trees:

                    is_dependent = len(tree) > 1  # and MODE != "BAB"

                    unexplained = np.ones((480, 640), dtype=np.uint8) if is_dependent else None
                    fixed = []

                    for i, hi in enumerate(tree):
                        obj_hypotheses = hypotheses[hi - 1]

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
                        renderer.set_observation(scene_depth.reshape(480, 640, 1), scene_normals.reshape(480, 640, 3))

                        # TODO check if len(tree) > 1 and i != len(tree)-1 -> only if dependencies are physical, not for occlusion!
                        Verefine.fit_fn = Verefine.fit_multi if len(
                            tree) > 1 else Verefine.fit_single  # TODO or by mask overlap? is_dependent does not work well...
                        pir.fixed = fixed  # note: s.t. sim in initial BAB scoring is correct
                        bab = BudgetAllocationBandit(pir, observation, obj_hypotheses, unexplained=unexplained)
                        bab.refine_max(fixed=fixed, unexplained=unexplained)
                        hypothesis, plays, fit = bab.get_best()
                        assert hypothesis is not None
                        if hypothesis.confidence < 0.1:
                            final_hypotheses.append(hypothesis)
                            refinements += bab.max_iter - len(obj_hypotheses)  # don't count initial render
                            continue
                        if is_dependent:
                            h_depth = hypothesis.render(observation, 'depth')[1] * 1000
                            # unexplained[np.logical_and(np.abs(h_depth - scene_depth) < 8, h_depth > 0)] = 0
                            # h_depth[h_depth - scene_depth > 5] = 0
                            h_depth[np.abs(h_depth - scene_depth) > 8] = 0
                            unexplained[h_depth > 0] = 0
                            obj_depths[hi - 1] = h_depth.copy()
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
                        final_hypotheses.append(None)
                        continue

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
                            final_hypotheses += [None, None]
                            continue
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

                                # plt.imshow(np.logical_and(unique_mask == 0, obj_depth > 0))  # TODO debug

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
                                                         scene_normals.reshape(480, 640, 3))

                                Verefine.fit_fn = Verefine.fit_multi

                                if babs[i] is None:
                                    pir.fixed = fixed  # note: s.t. sim in initial BAB scoring is correct
                                    babs[i] = SceneBAB(pir, observation, obj_hypotheses)

                                hi, h_sel, fit = babs[i].refine(fixed)

                                # update mask per object
                                h_depth = h_sel.render(observation, 'depth')[1] * 1000
                                h_depth[np.abs(h_depth - d) > 8] = 0  # TODO use per-pixel-fitness for this selection?
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
                            rendered = renderer.render(sel_ids, sel_trafos, camera_extrinsics, camera_intrinsics,
                                                       mode='depth+normal')
                            reward = Verefine.fit_single(observation, rendered, None)
                            # reward = Verefine.fit_scene(observation, rendered)  # TODO or some global fit fn?

                            # backprop
                            for i, (hi, _, _) in enumerate(selected):
                                babs[i].backpropagate(hi, reward)

                        # get best for final estimate
                        for i in range(len(tree)):
                            # hypothesis, _, _ = babs[i].get_best()  # TODO adapt to be based on reward
                            # hypothesis = babs[i].best
                            hypothesis = babs[i].pool[np.argmax([babs[i].rewards])][-1]
                            final_hypotheses.append(hypothesis)

                        refinements += scene_level_iter * len(tree)


            durations.append(time.time() - st)

            if len(renderer.runtimes) > 0:  # MODE not in ["BASE", "PIR"]:
                print("   ~ rendering/call=%0.1fms" % (np.mean(np.sum(renderer.runtimes, axis=1)) * 1000))
                print("   ~ cost/call=%0.1fms" % (np.mean(Verefine.cost_durations) * 1000))
            if len(renderer.runtimes) > 0:  # MODE not in ["BASE", "PIR"]:
                print("   ~ rendering/frame=%ims" % (np.sum(np.sum(renderer.runtimes, axis=1)) * 1000))
                print("   ~ cost/frame=%ims" % (np.sum(Verefine.cost_durations) * 1000))
            renderer.runtimes = []

            if PLOT:
                def dbg():
                    all_ren = np.zeros_like(rgb)
                    for hi, h in enumerate(final_hypotheses):
                        print("%s, %0.3f" % (obj_names[int(h.model)], h.confidence))
                        cur = h.render(observation, 'color')[0]
                        all_ren[all_ren==0] = cur[all_ren==0]
                    plt.imshow(all_ren/255*0.7 + rgb/255*0.3)
                dbg()
                plt.show()

            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(renderer.render([scene], [pose], camera_extrinsics, camera_intrinsics,
            #                            mode='color')[0]/255*0.7 + rgb/255*0.3)
            # plt.title("GT")
            # plt.subplot(1, 2, 2)
            # plt.imshow(final_hypotheses[0].render(observation, 'color')[0]/255*0.7 + rgb/255*0.3)
            # plt.title("refined")
            # plt.show()

            # write results
            if not PLOT:
                for hypothesis in final_hypotheses:
                    # with open("/home/dominik/projects/hsr-grasping/break/GT_lm-test.csv",
                    #           'a') as file:
                    # with open("/home/dominik/projects/hsr-grasping/break/%s%s%s%0.2d_lm-test.csv"
                    with open("/home/dominik/projects/hsr-grasping/log/%s/%sdf_ycbv-test.csv"
                              % (MODE, "" if MODE != "BAB" else "%i-" % Verefine.HYPOTHESES_PER_OBJECT), 'a') as file:
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
