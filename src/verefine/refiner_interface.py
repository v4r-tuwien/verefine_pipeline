import abc
import numpy as np

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from object_detector_msgs.msg import BoundingBox, Detection, PoseWithConfidence
from object_detector_msgs.srv import refine_poses, refine_posesResponse
import ros_numpy

from scipy.spatial.transform.rotation import Rotation

import src.verefine.verefine as Verefine  # TODO only needed until rendering fix
from src.verefine.verefine import Hypothesis, PhysIR, BudgetAllocationBandit
from src.verefine.simulator import Simulator
from src.verefine.plane_segmentation import PlaneDetector
from src.util.renderer import Renderer


class Refiner:

    def __init__(self, intrinsics, dataset, mode="base"):
        self.intrinsics = intrinsics
        self.dataset = dataset
        self.mode = mode  # base, pir, bab

        if mode == "pir" or "bab":
            self.plane_detector = PlaneDetector(640, 480, intrinsics, down_scale=4)
            self.simulator = Simulator(dataset, instances_per_object=5)  # TODO match with num hypotheses
            self.pir = PhysIR(self, self.simulator)
        if mode == "bab":
            self.renderer = Renderer(dataset)

    @abc.abstractmethod
    def refine(self, rgb, depth, intrinsics, obj_roi, obj_mask, obj_id,
               estimate, iterations):
        pass

    def ros_refine(self, req):
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

        iterations = 2  # TODO parameter

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
            self.renderer.create_egl_context()

            # prepare observation for object-level verification
            observation = {
                "depth": depth,
                "extrinsics": extrinsics,
                "intrinsics": self.intrinsics
            }

            # to our hypothesis format
            hypotheses = [Refiner.ros_estimate_to_hypothesis(estimate, obj_id, i_est)
                          for i_est, estimate in enumerate(estimates)]

            # ...
            refiner_params = []
            for estimate in estimates:
                estimate = [
                    ros_numpy.numpify(estimate.pose.orientation),
                    ros_numpy.numpify(estimate.pose.position),
                    estimate.confidence
                ]
                refiner_params.append((rgb, depth, self.intrinsics, obj_roi, obj_mask, obj_id, estimate, iterations))

            # refine
            Verefine.RENDERER = self.renderer  # TODO more elegant solution? -> adapt BAB and SV implementation
            bab = BudgetAllocationBandit(self.pir, observation, hypotheses)  # TODO define max iter here
            bab.refine_max(refiner_params)
            hypothesis, plays, fit = bab.get_best()
            assert hypothesis is not None

            # back to interface's format
            hypothesis = [Rotation.from_dcm(np.array(hypothesis.transformation[:3, :3])).as_quat(),
                          np.array(hypothesis.transformation[:3, 3]).reshape(3),
                          fit]
            refined = [hypothesis]
        else:
            refined = []
            for i_est, estimate in enumerate(estimates):
                if self.mode == "base":
                    estimate = [
                        ros_numpy.numpify(estimate.pose.orientation),
                        ros_numpy.numpify(estimate.pose.position),
                        estimate.confidence
                    ]

                    hypothesis = self.refine(rgb, depth, self.intrinsics, obj_roi, obj_mask, obj_id,
                                             estimate, iterations)
                elif self.mode == "pir":
                    # to our hypothesis format
                    hypothesis = Refiner.ros_estimate_to_hypothesis(estimate, obj_id, i_est)

                    # refine
                    estimate = [
                        ros_numpy.numpify(estimate.pose.orientation),
                        ros_numpy.numpify(estimate.pose.position),
                        estimate.confidence
                    ]
                    refiner_params = (rgb, depth, self.intrinsics, obj_roi, obj_mask, obj_id, estimate, iterations)
                    hypothesis = self.pir.refine(hypothesis, refiner_params)[-1]  # take final result

                    # back to interface's format
                    hypothesis = [Rotation.from_dcm(np.array(hypothesis.transformation[:3, :3])).as_quat(),
                                  np.array(hypothesis.transformation[:3, 3]).reshape(3),
                                  hypothesis.confidence]  # TODO should we change it based on some metric of PhysIR?
                else:
                    raise ValueError("unknown mode %s" % self.mode)
                refined.append(hypothesis)
            assert len(refined) == len(estimates)

        # === OUT ===
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
