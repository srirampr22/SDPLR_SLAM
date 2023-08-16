import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the pcl_mat_numpy of size (N,6) (column: x=0, y=1, z=2, r=3 ,g=4, b=5)
pcl_mat = np.load('converted_points2.npy')

pcl_mat = pcl_mat[~np.isnan(pcl_mat).any(axis=1)]

print(pcl_mat.shape)

# # Separate the position and color data
positions = pcl_mat[:, :3]  # XYZ coordinates
colors = pcl_mat[:, 3:]  # RGB colors



# # Matplotlib figure and an Axes3D object
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=1)

# # labels for the axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()
# Separate the position and color data
# Separate the position and color data
# positions = pcl_mat[:, :3]  # XYZ coordinates
# colors = np.repeat(np.array([[1, 0, 0]]), positions.shape[0], axis=0)  # Create color array with (1, 0, 0) for red

# Matplotlib figure and an Axes3D object
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=1)

# labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()