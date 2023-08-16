#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pickle

# Load the data from the file
with open('position_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract the variables from the loaded data
x_positions = data['x_positions']
y_positions = data['y_positions']
z_positions = data['z_positions']
timestamps = list(range(len(x_positions)))

# Plot x, y, and z positions vs. time as line graphs
plt.plot( x_positions, timestamps, label='X', color='red', marker='o')
# plt.plot(timestamps, y_positions, label='Y', color='green', linestyle='-')
# plt.plot(timestamps, z_positions, label='Z', color='blue', linestyle='-')

plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('X, Y, Z Positions vs. Time')
plt.legend()
plt.grid(True)
plt.show()


