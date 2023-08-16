#!/usr/bin/env python3

import numpy as np
import pandas as pd
import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyntcloud import PyntCloud

# Load the points from the 'transformed_points.csv' file
file_path = 'ref_mean_cov/48_inch_mean_cov.csv'
points = np.loadtxt(file_path, delimiter=',')
print(points.shape)

# Extract the RGB column
# rgb_values = points[:, 3].astype(int)

# # Unpack RGB values
# r = (rgb_values >> 16) & 255
# g = (rgb_values >> 8) & 255
# b = rgb_values & 255

# # Print the minimum and maximum of r, g, b
# print(f"Min and Max of r: {r.min()}, {r.max()}")
# print(f"Min and Max of g: {g.min()}, {g.max()}")
# print(f"Min and Max of b: {b.min()}, {b.max()}")

# Convert the points to a DataFrame
point_columns = ['x', 'y', 'z']  # Assuming your CSV file has 6 columns (x, y, z, red, green, blue)
df_points = pd.DataFrame(points[:,:3], columns=point_columns)


# Convert DataFrame to an open3d point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(df_points.values)

# Define the voxel size for downsampling (you can adjust this value as needed)
voxel_size = 0.0085

# Downsample the point cloud using voxel_down_sample
downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

# Convert the downsampled point cloud back to a DataFrame
downsampled_points = np.asarray(downsampled_pcd.points)
df_downsampled_points = pd.DataFrame(downsampled_points, columns=point_columns)

mean_point = np.mean(downsampled_points, axis=0)

# Calculate the covariance matrix of the downsampled points
cov_matrix = np.cov(downsampled_points, rowvar=False)

print("Mean of downsampled points:")
print(mean_point)

print("Covariance matrix of downsampled points:")
print(cov_matrix)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original points as a scatter plot
# ax.scatter(df_points['x'], df_points['y'], df_points['z'], c='b', marker='.', label='Transformed Points')

# Plot the downsampled points as a scatter plot
ax.scatter(df_downsampled_points['x'], df_downsampled_points['y'], df_downsampled_points['z'], c='r', marker='.', label='Downsampled Points')

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualized Transformed Points')

# Add a legend
ax.legend()

# Show the plot
plt.show()