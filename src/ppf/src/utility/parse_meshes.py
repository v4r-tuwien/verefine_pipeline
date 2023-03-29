import os
import open3d as o3d
import numpy as np
import glob

voxel_size = 0.005  # in meters

models_path = "/canister/data/models"
for model_path in glob.glob(os.path.join(models_path, "*.ply")) + glob.glob(os.path.join(models_path, "*.stl")):
    model_name = model_path.split("/")[-1][:-4]  # assumes file ending with 4 characters (incl '.')
    model_format = model_path.split("/")[-1][-4:]
    if os.path.exists(os.path.join(models_path, f"{model_name}")):
        print(f"Model {model_name} ({model_path}) already exists. Delete folder models/{model_name} to recompute.")
        continue
    print(f"Converting model {model_name} ({model_path})...")

    if model_format == ".ply":  # load in pcd format
        pcd = o3d.io.read_point_cloud(model_path)
    elif model_format == ".stl":  # load mesh, convert to pcd
        mesh = o3d.io.read_triangle_mesh(model_path)
        # scale to mm if needed
        if mesh.get_max_bound().max() > 1.0:  # would be > 1m --> assume this is actually in mm, scale to meters
            mesh = mesh.scale(1e-3, [0, 0, 0])
        # add normals if missing
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        # sample surface
        pcd = mesh.sample_points_uniformly(5000)

    # get a lower resolution pcd
    pcd = pcd.voxel_down_sample(voxel_size)
    # add colors if missing
    if not pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcd.points), 3)) * 0.5)

    # make directory for model name
    if not os.path.exists(os.path.join(models_path, f"{model_name}")):
        os.mkdir(os.path.join(models_path, f"{model_name}"))
    o3d.io.write_point_cloud(os.path.join(models_path, f"{model_name}/3D_model.pcd"), pcd)
    # make directory for views
    if not os.path.exists(os.path.join(models_path, f"{model_name}/views")):
        os.mkdir(os.path.join(models_path, f"{model_name}/views"))
    o3d.io.write_point_cloud(os.path.join(models_path, f"{model_name}/views/cloud_0.pcd"), pcd)
# PPF will generate a hash per model upon first run
