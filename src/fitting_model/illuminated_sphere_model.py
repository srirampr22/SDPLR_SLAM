import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_illuminated_sphere(radius, center, light_dir):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Create a mask for the illuminated portion of the sphere.
    # We will only keep the points where the z coordinate is greater than the light source's z coordinate.
    mask = np.dot(np.array([x, y, z]).T - center, light_dir) > 0


    return x[mask], y[mask], z[mask]

def main():
    radius = 1.0
    center = [0.0, 0.0, 0.0]
    light_dir = np.array([0.0, 1.0, 0.0])  # Light coming from above (positive z direction)

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Create a mask for the illuminated portion of the sphere
    mask = np.dot(np.array([x.flatten(), y.flatten(), z.flatten()]).T - center, light_dir) > 0

    x = x.flatten()[mask]
    y = y.flatten()[mask]
    z = z.flatten()[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()

if __name__ == "__main__":
    main()
