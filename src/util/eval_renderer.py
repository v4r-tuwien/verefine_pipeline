# import matplotlib
# matplotlib.use("Qt5Agg")
# import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.io as scio
from scipy.spatial.transform.rotation import Rotation
import PIL
import time

from src.util.fast_renderer import Renderer
from src.util.dataset import YcbvDataset


PATH_BOP19 = "/mnt/Data/datasets/BOP19/"
PATH_YCBV = "/mnt/Data/datasets/YCB Video/YCB_Video_Dataset/"

SEGMENTATION = "GT"
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


def render(obj_ids, transformations, extrinsics, intrinsics, mode):
    # assert mode in ['color', 'depth', 'depth+seg', 'color+depth+seg']

    return renderer.render(obj_ids, transformations, extrinsics, intrinsics, mode=mode)


# -----------------

if __name__ == "__main__":

    dataset = YcbvDataset(base_path=PATH_YCBV)

    renderer = Renderer(dataset)
    # renderer.create_egl_context()
    durations = []

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
    scenes = [48]  # sorted(np.unique(scene_ids))
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

            # meta data (PCNN segmentation and poses
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
            camera_extrinsics[:3, :3] = np.matrix(frame_camera["cam_R_w2c"]).reshape(3, 3).T
            camera_extrinsics[:3, 3] = (-np.matrix(frame_camera["cam_R_w2c"]).reshape(3, 3).T * np.matrix(frame_camera["cam_t_w2c"]).T) / 1000.0

            camera_intrinsics = np.array(frame_camera["cam_K"]).reshape(3, 3)

            # get pose estimates
            obj_ids_ = [int(v) for v in meta['rois'][:, 1]]
            obj_poses = meta['poses_icp']
            obj_Ts = []
            for obj_id, obj_pose in zip(obj_ids_, obj_poses):
                obj_q = obj_pose[:4]
                obj_pose[:3] = obj_q[1:]
                obj_pose[3] = obj_q[0]

                obj_T = np.matrix(np.eye(4))
                obj_T[:3, :3] = Rotation.from_quat(obj_pose[:4]).as_dcm()
                obj_T[:3, 3] = obj_pose[4:].reshape(3, 1)

                obj_confidence = 1.0

                obj_Ts.append(obj_T)

            # render
            st = time.time()
            renderer.set_observation(depth.reshape(480, 640, 1))
            mode = "cost"
            # rendered = render([obj_ids_[0]], [obj_Ts[0]], camera_extrinsics, camera_intrinsics, mode=mode)  # single object
            rendered = render(obj_ids_, obj_Ts, camera_extrinsics, camera_intrinsics, mode=mode)  # multi object
            if mode == "cost":
                rendered, fit = rendered

            durations.append(time.time() - st)

            # plt.imshow(rendered[0])
            # plt.show()
    print("total = %0.1fms" % (np.mean(durations)*1000))
    print("total (w/o first) = %0.1fms" % (np.mean(durations[1:])*1000))
    print("---")
    print("draw = %0.1fms" % (np.mean(renderer.runtimes, axis=0)[0]*1000))
    print("read = %0.1fms" % (np.mean(renderer.runtimes, axis=0)[1]*1000))
    print("cost = %0.1fms" % (np.mean(renderer.runtimes, axis=0)[2]*1000))
