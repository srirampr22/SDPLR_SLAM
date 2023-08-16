# Import the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate point cloud of a sphere
def generate_sphere(radius, center, n_points):
    # Generate random angles
    theta = 2.0 * np.pi * np.random.rand(n_points)  # polar angle
    phi = np.arccos(2.0 * np.random.rand(n_points) - 1.0)  # azimuthal angle

    # Convert the spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]

    return x, y, z


if __name__ == "__main__":
    # parameters of the sphere
    radius = 1.0  # Radius of the sphere
    center = np.array([0.0, 0.0, 0.0])  # Center of the sphere

    # number of points to generate
    n_points = 1000

    # Generate the sphere point cloud
    x, y, z = generate_sphere(radius, center, n_points)

    # 3D plot of the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='b', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    points = np.vstack((x, y, z))

    sphere_pcl_mat = np.column_stack((x, y, z))

    # Calculate the mean of the points
    model_mean = np.mean(sphere_pcl_mat, axis=0)

    # Calculate the covariance matrix
    model_cov = np.cov(sphere_pcl_mat, rowvar=False)

    print("Model Mean of point cloud: ", model_mean)
    print("Model Covariance matrix: ", model_cov)
    plt.show()

    # main()
