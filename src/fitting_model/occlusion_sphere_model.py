import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_point_cloud(radius, center, n_points):
    # Generate random points within the cube of side 2*radius and then only keep points within the sphere
    points = np.random.rand(n_points, 3) * 2 * radius - radius
    points += center
    mask = np.linalg.norm(points - center, axis=1) <= radius
    return points[mask]

def visible_points(points, observer):
    center = np.mean(points, axis=0)
    if observer[0] == center[0]:
        mask = points[:,2] <= observer[2]
    else:
        mask = points[:,2] <= ((observer[2]-center[2])/(observer[0]-center[0]))*(points[:,0]-center[0]) + center[2]
    return points[mask]

def main():
    radius = 10
    center = np.array([0, 0, 0])
    observer = np.array([0, 0, 0])

    points = generate_point_cloud(radius, center, n_points=10000)
    visible = visible_points(points, observer)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(visible[:,0], visible[:,1], visible[:,2])
    plt.show()

if __name__ == '__main__':
    main()
