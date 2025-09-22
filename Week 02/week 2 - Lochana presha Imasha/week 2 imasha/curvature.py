import pyvista as pv
import numpy as np

# 1. Load the mesh (use your correct path)
mesh = pv.read(r"I:\Thingi10K\raw_meshes\32770.stl")

# 2. Compute curvature
gaussian_curvatures = mesh.curvature(curv_type='gaussian')

# 3. Print and analyze the values
print("The first 10 curvature values are:")
print(gaussian_curvatures[:10])

print("\n--- Curvature Statistics ---")
print(f"Minimum curvature: {np.min(gaussian_curvatures):.4f}")
print(f"Maximum curvature: {np.max(gaussian_curvatures):.4f}")
print(f"Average curvature: {np.mean(gaussian_curvatures):.4f}")
print(f"Number of values: {len(gaussian_curvatures)}")

# 4. (Optional) visualize
p = pv.Plotter()
p.add_mesh(mesh, scalars=gaussian_curvatures, show_scalar_bar=True)
p.show()
