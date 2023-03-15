import open3d as o3d
import os
import re
import numpy as np
import itertools


dir_path = "/home/christian/v4r/verefine_project/verefine_pipeline/data/models/render"
dir_path2 = "/home/christian/v4r/verefine_project/verefine_pipeline/data/models/renderer"

filename = "final_hand.ply"
files = []
folders = []
for path in sorted(os.listdir(dir_path)):
    # check if current path is a file
    if re.search(".ply", path):
        files.append(path)
        
for path in sorted(os.listdir(dir_path2)):
    # check if current path is a file
    folders.append(path)
        
print(files)
print(folders)

for index in range(0, len(files)):
    mesh = o3d.io.read_triangle_mesh(dir_path + "/" + files[index])
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(dir_path2 + "/" + folders[index] + ".stl", mesh)
        

