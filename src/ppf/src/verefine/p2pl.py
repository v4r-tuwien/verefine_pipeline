import numpy as np
import open3d as o3d
from copy import deepcopy


def refine(pose, cloud_observed, cloud_model, p_distance=0.1, iterations=1, p2pl=True):
    obj_T = pose.copy()  # will be altered
    source = deepcopy(cloud_observed)  # will be altered
    target = cloud_model

    # get scale to adapt p_distance (inlier threshold)
    s = np.linalg.norm(target.get_max_bound() - target.get_min_bound()) / 2
    p_distance *= s  # p_distance is percentage of max distance -> to absolute value based on scale

    # refine
    if p2pl:  # Point-to-Plane
        T = o3d.pipelines.registration.registration_icp(
            source.transform(np.linalg.inv(obj_T)), target, p_distance, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        ).transformation
    else:  # Point-to-Point
        T = o3d.pipelines.registration.registration_icp(
            source.transform(np.linalg.inv(obj_T)), target, p_distance, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        ).transformation

    obj_T = obj_T @ np.linalg.inv(T)
    return obj_T
