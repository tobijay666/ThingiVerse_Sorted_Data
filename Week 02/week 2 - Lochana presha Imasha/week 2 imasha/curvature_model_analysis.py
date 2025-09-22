# curvature_model_analysis.py

import trimesh
import numpy as np

# -----------------------------
# 1. Load the STL mesh
# -----------------------------
mesh_path = r"I:\Thingi10K\raw_meshes\32770.stl"  # change to your STL path
mesh = trimesh.load_mesh(mesh_path)
print(f"Mesh loaded: {mesh_path}")
print(f"Number of vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")

# -----------------------------
# 2. Compute Gaussian curvature (angle deficit)
# -----------------------------
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

# -----------------------------
# 3. Print curvature stats
# -----------------------------
min_curv = np.min(curvatures)
max_curv = np.max(curvatures)
mean_curv = np.mean(curvatures)

print("\n--- Curvature Statistics ---")
print("First 10 curvature values:", curvatures[:10])
print(f"Minimum curvature: {min_curv:.6f}")
print(f"Maximum curvature: {max_curv:.6f}")
print(f"Average curvature: {mean_curv:.6f}")

# -----------------------------
# 4. Decide model type
# -----------------------------
# Define threshold for flat vs free-form
threshold = 1e-3  # adjust if needed
if max_curv < threshold:
    model_type = "Flat Model"
else:
    model_type = "Free-form Model"

print(f"\n--- Model Type ---\nThis model is classified as: {model_type}")

# -----------------------------
# 5. Optional: Visualize with PyVista
# -----------------------------
try:
    import pyvista as pv
    # Convert trimesh faces to PyVista format
    faces_pv = np.hstack((np.full((len(mesh.faces),1),3), mesh.faces)).astype(np.int64)
    mesh_pv = pv.PolyData(mesh.vertices, faces_pv)
    
    # Normalize curvature for coloring
    curv_norm = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures) + 1e-12)
    
    p = pv.Plotter()
    p.add_mesh(mesh_pv, scalars=curv_norm, show_scalar_bar=True)
    p.add_axes()
    p.show()
except ModuleNotFoundError:
    print("\nPyVista not installed. Skipping visualization. To see 3D plot, run: pip install pyvista pyvistaqt")
