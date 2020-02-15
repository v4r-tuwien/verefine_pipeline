import abc
import numpy as np

import torch
import gc

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from object_detector_msgs.msg import BoundingBox, Detection, PoseWithConfidence
from object_detector_msgs.srv import refine_poses, refine_posesResponse
import ros_numpy

import cv2 as cv
from scipy.spatial.transform.rotation import Rotation

import src.verefine.verefine as Verefine  # TODO only needed until rendering fix
from src.verefine.verefine import Hypothesis, PhysIR, BudgetAllocationBandit
from src.verefine.simulator import Simulator
from src.verefine.plane_segmentation import PlaneDetector
from src.util.fast_renderer import Renderer


class Refiner:

    def __init__(self, intrinsics, dataset, mode="base"):
        self.intrinsics = intrinsics
        self.dataset = dataset
        self.mode = mode  # base, pir, bab

        #if mode == "pir" or "bab":
        self.plane_detector = PlaneDetector(640, 480, intrinsics, down_scale=4)
        self.simulator = Simulator(dataset, instances_per_object=5)  # TODO match with num hypotheses
        self.pir = PhysIR(self, self.simulator)
        #if mode == "bab":
        self.renderer = Renderer(dataset)

    @abc.abstractmethod
    def refine(self, rgb, depth, intrinsics, obj_roi, obj_mask, obj_id,
               estimate, iterations):
        pass

    def ros_refine(self, req):
        print("refinement requested...")

        # === IN ===
        # --- rgb
        rgb = req.rgb
        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480
        rgb = ros_numpy.numpify(rgb)

        # --- depth
        depth = req.depth
        depth = ros_numpy.numpify(depth)

        # --- detection

        # obj name to id
        name = req.det.name
        obj_id = -1
        for idx, obj_name in self.dataset.obj_names.items():
            if obj_name == name:
                obj_id = idx + 1
                break
        assert obj_id > 0  # should start from 1

        # parse roi
        bbox = req.det.bbox
        obj_roi = [bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax]

        # mask indices to image-sized binary mask
        mask_ids = np.array(req.det.mask)
        obj_mask = np.zeros((height * width), dtype=np.uint8)
        obj_mask[mask_ids] = 1
        obj_mask = obj_mask.reshape((height, width))

        # --- estimates
        estimates = req.poses

        # === POSE REFINEMENT ===
        refined = []
        gc.collect()
        torch.cuda.empty_cache()

        # check mode
        import json
        with open("/densefusion/src/config.json", 'r') as file:
            config = json.load(file)
        self.mode = config["refinement"]["mode"]
        num_hypotheses = config["estimation"]["num_hypotheses"]
        Verefine.HYPOTHESES_PER_OBJECT = num_hypotheses
        iterations = config["refinement"]["num_iterations"]
        Verefine.ITERATIONS = iterations
        sim_steps = config["refinement"]["num_sim_steps"]
        Verefine.SIM_STEPS
        Verefine.C = config["refinement"]["bab_c"]

        print("   refining: mode=%s, iterations=%i, sim_steps=%i..." % (self.mode, iterations, sim_steps))

        # some modes require extrinsics of cam w.r.t. scene ground plane (via plane segmentation)
        if self.mode in ["pir", "bab"]:
            observation_mask = obj_mask  # TODO merge masks for multiple detections
            try:
                extrinsics = self.plane_detector.detect(depth, observation_mask)
            except ValueError as e:
                print(e)
                # TODO fallback to extrinsics of camera w.r.t. world?

                pose = PoseWithConfidence()
                pose.name = "invalid"
                pose.confidence = -1

                response = refine_posesResponse()
                response.poses = [pose]
                return response
            self.simulator.initialize_frame(extrinsics)

        # refinement according to selected mode
        if self.mode == "bab":

            # fix for multi-threading
            #self.renderer.create_egl_context()
            scene_depth = depth.copy()
            scene_depth[obj_mask == 0] = 0


            def estimate_normals(D):
                camera_intrinsics = self.intrinsics
                D_px = D.copy() * camera_intrinsics[0, 0]  # from meters to pixels

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

            #print("   estimate normals...")
            scene_normals = estimate_normals(scene_depth / 1000)

            # prepare observation for object-level verification
            observation = {
                "depth": scene_depth,
                "normals": scene_normals,
                "extrinsics": extrinsics,
                "intrinsics": self.intrinsics,
                "mask_others": np.zeros_like(depth)
            }

            # to our hypothesis format
            hypotheses = [Refiner.ros_estimate_to_hypothesis(estimate, obj_id, i_est)
                          for i_est, estimate in enumerate(estimates)]

            # ...
            for estimate, hypothesis in zip(estimates, hypotheses):
                estimate = [
                    ros_numpy.numpify(estimate.pose.orientation),
                    ros_numpy.numpify(estimate.pose.position),
                    estimate.confidence
                ]
                refiner_params = [rgb, depth, self.intrinsics, obj_roi, obj_mask, obj_id, estimate,
                                             Verefine.ITERATIONS, None, None, None]
                hypothesis.refiner_param = refiner_params


            # refine # TODO called once per object!
            #print("   set observation...")
            Verefine.RENDERER = self.renderer  # TODO more elegant solution? -> adapt BAB and SV implementation
            self.renderer.set_observation(scene_depth.reshape(480,640,1), scene_normals)
            #print("   run BAB...")
            try:
                #print("      1) init bab with %i hypotheses" % len(hypotheses))
                bab = BudgetAllocationBandit(self.pir, observation, hypotheses)  # TODO define max iter here
                #print("      2) refining...")
                bab.refine_max()
                #print("      3) get best estimate")
                hypothesis, plays, fit = bab.get_best()
                assert hypothesis is not None
                #print("   returning refined estimates...")
                # back to interface's format
                hypothesis = [Rotation.from_dcm(np.array(hypothesis.transformation[:3, :3])).as_quat(),
                            np.array(hypothesis.transformation[:3, 3]).reshape(3),
                            hypothesis.confidence]
                refined = [hypothesis]
            except Exception as ex:
                print(ex)
        else:
            for i_est, estimate in enumerate(estimates):
                if self.mode == "base":
                    estimate = [
                        ros_numpy.numpify(estimate.pose.orientation),
                        ros_numpy.numpify(estimate.pose.position),
                        estimate.confidence
                    ]

                    hypothesis = self.refine(rgb, depth, self.intrinsics, obj_roi, obj_mask, obj_id,
                                             estimate, Verefine.ITERATIONS)
                elif self.mode == "pir":
                    # to our hypothesis format
                    hypothesis = Refiner.ros_estimate_to_hypothesis(estimate, obj_id, i_est)

                    # refine
                    estimate = [
                        ros_numpy.numpify(estimate.pose.orientation),
                        ros_numpy.numpify(estimate.pose.position),
                        estimate.confidence
                    ]
                    #refiner_params = (rgb, depth, self.intrinsics, obj_roi, obj_mask, obj_id, estimate, iterations)
                    refiner_params = [rgb, depth, self.intrinsics, obj_roi, obj_mask, obj_id, estimate,
                                             Verefine.ITERATIONS, None, None, None]
                    hypothesis.refiner_param = refiner_params
                    hypothesis = self.pir.refine(hypothesis)[-1]  # take final result

                    # back to interface's format
                    hypothesis = [Rotation.from_dcm(np.array(hypothesis.transformation[:3, :3])).as_quat(),
                                  np.array(hypothesis.transformation[:3, 3]).reshape(3),
                                  hypothesis.confidence]  # TODO should we change it based on some metric of PhysIR?
                else:
                    raise ValueError("unknown mode %s" % self.mode)
                refined.append(hypothesis)
            assert len(refined) == len(estimates)

        # === OUT ===
        print("   responding...")
        poses = []
        for r, t, c in refined:
            pose = PoseWithConfidence()

            # --- name
            pose.name = name

            # --- pose
            pose.pose = Pose()
            pose.pose.position = ros_numpy.msgify(Point, t)
            pose.pose.orientation = ros_numpy.msgify(Quaternion, r)

            # --- confidence
            pose.confidence = c

            poses.append(pose)

        response = refine_posesResponse()
        response.poses = poses
        return response

    @staticmethod
    def ros_estimate_to_hypothesis(estimate, obj_id, instance=0):
        estimate = [
            ros_numpy.numpify(estimate.pose.orientation),
            ros_numpy.numpify(estimate.pose.position),
            estimate.confidence
        ]

        T_obj = np.matrix(np.eye(4))
        T_obj[:3, :3] = Rotation.from_quat(estimate[0]).as_dcm()
        T_obj[:3, 3] = estimate[1].reshape(3, 1)
        hypothesis = Hypothesis("%0.2d" % obj_id, T_obj,
                                roi=None, mask=None,  # TODO should we add those? needed?
                                embedding=None, cloud=None,  # TODO remove from Hypothesis?
                                instance_id=instance, confidence=estimate[2])
        return hypothesis
