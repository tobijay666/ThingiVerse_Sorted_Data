# batch_curvature_classification.py

import os
import trimesh
import numpy as np

# -----------------------------
# 1. Set folder path containing STL files
# -----------------------------
folder_path = r"I:\Thingi10K\raw_meshes"  # change to your folder

# Threshold for flat vs free-form (adjust if needed)
curvature_threshold = 1e-3

# Optional: PyVista visualization
use_visualization = False
try:
    import pyvista as pv
    use_visualization = True
except ModuleNotFoundError:
    print("PyVista not installed. Visualization will be skipped.")

# -----------------------------
# 2. Function to compute curvature
# -----------------------------
def compute_gaussian_curvature(mesh):
    curvatures = np.zeros(len(mesh.vertices))
    for i, vertex in enumerate(mesh.vertices):
        faces_idx = np.where(mesh.faces == i)[0]
        angle_sum = 0.0
        for f in faces_idx:
            face = mesh.faces[f]
            other_idx = [idx for idx in face if idx != i]
            v0 = mesh.vertices[other_idx[0]] - vertex
            v1 = mesh.vertices[other_idx[1]] - vertex
            cos_angle = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angle_sum += angle
        curvatures[i] = 2 * np.pi - angle_sum
    return curvatures

# -----------------------------
# 3. Process all STL files in folder
# -----------------------------
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".stl"):
        file_path = os.path.join(folder_path, filename)
        mesh = trimesh.load_mesh(file_path)
        curvatures = compute_gaussian_curvature(mesh)
        max_curv = np.max(curvatures)

        # Classify model
        model_type = "Flat Model" if max_curv < curvature_threshold else "Free-form Model"
        print(f"{filename}: {model_type}")

        # Optional visualization of the first file
        if use_visualization and filename == os.listdir(folder_path)[0]:
            faces_pv = np.hstack((np.full((len(mesh.faces),1),3), mesh.faces)).astype(np.int64)
            mesh_pv = pv.PolyData(mesh.vertices, faces_pv)
            curv_norm = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures) + 1e-12)
            p = pv.Plotter()
            p.add_mesh(mesh_pv, scalars=curv_norm, show_scalar_bar=True)
            p.add_axes()
            p.show()
