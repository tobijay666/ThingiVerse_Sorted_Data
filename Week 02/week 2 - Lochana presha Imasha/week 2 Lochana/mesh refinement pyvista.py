import pyvista as pv

# Load your original mesh
mesh = pv.read(r"C:\Users\user\Documents\1 code\Research\data set\free form\32770.stl")
print("Original Mesh Info:")
print(f"Number of Vertices: {mesh.n_points}")
print(f"Number of Faces: {mesh.n_cells}")

# Perform mesh refinement using the Loop subdivision algorithm
# nsub=1 means each face is split into 4 smaller faces.
refined_mesh = mesh.subdivide(nsub=1, subfilter='loop')

# Print the new, refined mesh's information
print("\nRefined Mesh Info:")
print(f"Number of Vertices: {refined_mesh.n_points}")
print(f"Number of Faces: {refined_mesh.n_cells}")

# Visualize the refined mesh to see the result
# This will open a new window showing the more detailed mesh.
refined_mesh.plot(show_edges=True, show_bounds=True)