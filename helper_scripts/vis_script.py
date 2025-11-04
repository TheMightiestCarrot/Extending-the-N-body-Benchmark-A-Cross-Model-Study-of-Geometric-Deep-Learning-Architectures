import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example: assume `positions` is a (timesteps, n_bodies, 3) numpy array for 3D positions
timesteps = 1000
n_bodies = 1500

true = np.load('runs/ponita/2024-11-12_14-58-17/checkpoints/89056/generated_trajectories/2024-11-12_23-45-46/trajectories_data/loc_actual_sim_0.npy')
pred = np.load('runs/ponita/2024-11-12_14-58-17/checkpoints/89056/generated_trajectories/2024-11-12_23-45-46/trajectories_data/loc_pred_sim_0.npy')
positions = np.concatenate([true, pred], axis=1)
# positions = np.random.rand(timesteps, n_bodies, 3)  # Replace with your actual data

# Define colors for the two groups
colors = np.zeros((n_bodies, 4), dtype=np.float32)  # RGBA colors
colors[:n_bodies // 2] = [1.0, 0.0, 0.0, 1.0]  # Red
colors[n_bodies // 2:] = [0.0, 0.0, 1.0, 1.0]  # Blue

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], c=colors, s=5)

# Set limits for the 3D view
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(-25, 25 )

# Function to update the scatter plot for each frame
def update(frame):
    ax.cla()  # Clear the axes
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-25, 25 )
    sc = ax.scatter(positions[frame, :, 0], positions[frame, :, 1], positions[frame, :, 2], c=colors, s=5)

# Create animation
ani = FuncAnimation(fig, update, frames=timesteps, repeat=False)


from matplotlib.animation import FFMpegWriter

# Use `libx264` explicitly and ensure pixel format compatibility
writer = FFMpegWriter(fps=30, extra_args=["-pix_fmt", "yuv420p"])
ani.save("simulation.mp4", writer=writer, dpi=100) 