import numpy as np
import json
from scipy.spatial.transform.rotation import Rotation
import PIL
import sys
sys.path.append("/home/dominik/projects/hsr-grasping")
sys.path.append("/home/dominik/projects/hsr-grasping/src")
sys.path.append("/home/dominik/projects/hsr-grasping/src/util")

from src.util.dataset import LmDataset


# settings
PATH_BOP19 = "/mnt/Data/datasets/BOP19/"
PATH_LM = "/mnt/Data/datasets/SIXD/LM_LM-O/"
PATH_LM_ROOT = '/mnt/Data/datasets/Linemod_preprocessed/'

SCENES = []

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

    t_errs, z_errs, r_errs = [], [], []

    dataset = LmDataset(base_path=PATH_LM)
    SCENES = dataset.objlist[1:]

    with open("/home/dominik/projects/hsr-grasping/log/BAB_top0.7/1-t05_lm-test.csv", 'r') as file:
        hypotheses = file.readlines()

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
    hi = 0
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

            # get obj ids, num objs, obj names
            frame_obj_ids = [scene]  #scene_obj_ids[frame_target_indices] if SEGMENTATION == "GT" else np.unique(labels)[1:]
            frame_num_objs = 1  # len(frame_obj_ids)
            frame_obj_names = [obj_names[int(idx)] for idx in frame_obj_ids]

            # get extrinsics and intrinsics
            frame_camera = scene_camera[str(frame)]

            camera_intrinsics = np.array(frame_camera["cam_K"]).reshape(3, 3)

            pose_info = scene_gt['%i' % frame][0]
            pose_gt = np.matrix(np.eye(4))
            pose_gt[:3, :3] = np.array(pose_info['cam_R_m2c']).reshape(3, 3)
            pose_gt[:3, 3] = np.array(pose_info['cam_t_m2c']).reshape(3, 1)/1000

            camera_extrinsics = pose_gt.copy()
            camera_extrinsics[:3, 3] = camera_extrinsics[:3, 3] + camera_extrinsics[:3, :3]*np.matrix([0.0, 0.0, dataset.obj_bot[scene-1]]).T

            hypothesis = hypotheses[hi]
            assert hypothesis.startswith("%0.2d,%i" % (scene, frame))
            hi += 1

            parts = hypothesis.split(",")
            pose_est = np.matrix(np.eye(4))
            pose_est[:3, :3] = np.array([float(v) for v in parts[4].split(" ")]).reshape(3, 3)
            pose_est[:3, 3] = np.array([float(v) for v in parts[5].split(" ")]).reshape(3, 1)/1000

            gt_in_world = camera_extrinsics.I * pose_gt
            est_in_world = camera_extrinsics.I * pose_est

            t_err = np.linalg.norm(gt_in_world[:3, 3] - est_in_world[:3, 3]) / (-2*dataset.obj_bot[scene-1])
            z_err = np.abs(gt_in_world[2, 3] - est_in_world[2, 3]) / (-2*dataset.obj_bot[scene-1])
            r_err = np.rad2deg(np.arccos(np.dot(Rotation.from_dcm(gt_in_world[:3, :3]).as_quat(),
                                                Rotation.from_dcm(est_in_world[:3, :3]).as_quat())))

            t_errs.append(t_err)
            z_errs.append(z_err)
            r_errs.append(r_err)

    #print("t-err [mm]:  mean=%0.3f, std=%0.3f, n=%i" % (np.mean(t_errs)*1000, np.std(t_errs)*1000, len(t_errs)))
    #print("z-err [mm]:  mean=%0.3f, std=%0.3f, n=%i" % (np.mean(z_errs)*1000, np.std(z_errs)*1000, len(z_errs)))
    print("t-err [%%]:  mean=%0.3f, std=%0.3f, n=%i" % (np.mean(t_errs) * 100, np.std(t_errs) * 100, len(t_errs)))
    print("z-err [%%]:  mean=%0.3f, std=%0.3f, n=%i" % (np.mean(z_errs) * 100, np.std(z_errs) * 100, len(z_errs)))
    print("r-err [deg]: mean=%0.3f, std=%0.3f, n=%i" % (np.mean(r_errs), np.std(r_errs), len(r_errs)))
