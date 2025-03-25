import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm


def find_latest_activations_file(base_dir="/root/vast/eric/vnl-ray/training/activations/"):
    """Find the most recent activations HDF5 file"""
    h5_files = glob.glob(os.path.join(base_dir, "activations_*.h5"))
    if not h5_files:
        return None
    return max(h5_files, key=os.path.getmtime)


def plot_intentions_3d(activations_file=None, show_targets=True, max_episodes=10):
    """
    Visualize intentions as 3D trajectories similar to the kinematics data visualization.

    Args:
        activations_file (str, optional): Path to activations HDF5 file. If None, uses latest file.
        show_targets (bool, optional): Whether to show target positions alongside trajectories.
        max_episodes (int, optional): Maximum number of episodes to plot.
    """
    if activations_file is None:
        activations_file = find_latest_activations_file()

    if not activations_file or not os.path.exists(activations_file):
        print("No activations HDF5 file found.")
        return

    print(f"Analyzing intentions from: {activations_file}")

    # Always load corresponding kinematics file for timesteps and targets
    kinematics_file = activations_file.replace("activations_", "kinematics_")
    if not os.path.exists(kinematics_file):
        print(f"Warning: No matching kinematics file found at {kinematics_file}")
        print("Will use internal timesteps instead.")
        kinematics_data = None
    else:
        print(f"Found matching kinematics file: {kinematics_file}")
        with pd.HDFStore(kinematics_file, "r") as kstore:
            if "/pose_and_trial_info" in kstore:
                kinematics_data = kstore["/pose_and_trial_info"]
                print(f"Loaded pose and trial info: {kinematics_data.shape} rows")
                print(
                    f"Episode range: {kinematics_data['episode_number'].min()} to {kinematics_data['episode_number'].max()}"
                )
            else:
                print("No pose_and_trial_info found in kinematics file")
                kinematics_data = None

    with pd.HDFStore(activations_file, "r") as store:
        # Look for the intentions data
        intention_keys = [k for k in store.keys() if "intentions_dist" in k]

        if not intention_keys:
            print("No intentions data found in the activations file.")
            print(f"Available keys: {store.keys()}")
            return

        intention_key = intention_keys[0]
        print(f"Using intentions data from: {intention_key}")

        # Load intentions data
        intentions_df = store[intention_key]

        # Check if we have neuron columns
        neuron_columns = [col for col in intentions_df.columns if "neuron" in col.lower()]
        if not neuron_columns or len(neuron_columns) < 3:
            print(f"Not enough dimensions in intentions data. Found columns: {intentions_df.columns}")
            return

        # For 3D plot we need exactly 3 dimensions
        x_col, y_col, z_col = neuron_columns[:3]

        # Create figure for 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Determine episode structure and link with kinematics data
        has_episode_col = "episode" in intentions_df.columns

        if has_episode_col:
            intentions_episodes = intentions_df["episode"].unique()
            print(f"Found {len(intentions_episodes)} episodes in intentions data")

            if kinematics_data is not None:
                kinematics_episodes = kinematics_data["episode_number"].unique()
                print(f"Found {len(kinematics_episodes)} episodes in kinematics data")

                # Find overlap between the two datasets
                common_episodes = sorted(set(intentions_episodes) & set(kinematics_episodes))
                print(f"Found {len(common_episodes)} common episodes between datasets")

                if len(common_episodes) == 0:
                    print("Warning: No common episodes between intentions and kinematics")
                    # Fall back to using internal episode/timestep structure
                    episodes = intentions_episodes[:max_episodes]
                    aligned_with_kinematics = False
                else:
                    episodes = common_episodes[:max_episodes]
                    aligned_with_kinematics = True
            else:
                episodes = intentions_episodes[:max_episodes]
                aligned_with_kinematics = False
        else:
            # No episode column in intentions data
            print("No episode column found in intentions data")
            episodes = range(min(max_episodes, 1))
            aligned_with_kinematics = False

        print(f"Plotting {len(episodes)} episodes of intention trajectories")

        # Generate a colormap for the episodes
        colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

        # Plot each episode's intentions
        for i, episode in enumerate(episodes):
            # Get intentions data for this episode
            if has_episode_col:
                episode_intentions = intentions_df[intentions_df["episode"] == episode]
                print(f"Episode {episode} has {len(episode_intentions)} intention timesteps")
            else:
                # If no episode column, use all data as single episode
                episode_intentions = intentions_df

            # Get the 3D coordinates
            x_values = episode_intentions[x_col].values
            y_values = episode_intentions[y_col].values
            z_values = episode_intentions[z_col].values

            # Get timesteps from kinematics if available
            if aligned_with_kinematics and kinematics_data is not None:
                # Extract only data for this episode
                episode_kinematics = kinematics_data[kinematics_data["episode_number"] == episode]
                if len(episode_kinematics) > 0:
                    # Use the actual timesteps from kinematics
                    timesteps = episode_kinematics["index"].values
                    print(f"Using actual timesteps from kinematics: {min(timesteps)} to {max(timesteps)}")

                    # Check if lengths match
                    if len(timesteps) != len(x_values):
                        print(
                            f"Warning: Timesteps length ({len(timesteps)}) doesn't match intentions length ({len(x_values)})"
                        )
                        print("Generating synthetic timesteps instead")
                        timesteps = np.arange(len(x_values))
                else:
                    timesteps = np.arange(len(x_values))
            else:
                # Generate synthetic timesteps
                timesteps = np.arange(len(x_values))
                print(f"Using synthetic timesteps from 0 to {len(x_values)-1}")

            # Create a colormap for this trajectory based on timestep
            points = ax.scatter(
                x_values,
                y_values,
                z_values,
                c=timesteps,
                cmap="viridis",
                alpha=0.8,
                s=30,
                label=f"Episode {episode}" if has_episode_col else f"Episode {i+1}",
            )

            # Draw lines connecting the points in sequence
            ax.plot(x_values, y_values, z_values, color=colors[i], alpha=0.5, linewidth=1)

            # Add arrows to show direction of movement
            stride = max(1, len(x_values) // 10)
            for j in range(0, len(x_values) - stride, stride):
                ax.quiver(
                    x_values[j],
                    y_values[j],
                    z_values[j],
                    x_values[j + stride] - x_values[j],
                    y_values[j + stride] - y_values[j],
                    z_values[j + stride] - z_values[j],
                    color=colors[i],
                    alpha=0.8,
                    arrow_length_ratio=0.1,
                    normalize=True,
                )

            # Add start and end markers
            ax.scatter(x_values[0], y_values[0], z_values[0], color="green", s=100, marker="o", alpha=1)
            ax.scatter(x_values[-1], y_values[-1], z_values[-1], color="red", s=100, marker="x", alpha=1)

            # Plot the target if available
            if kinematics_data is not None and aligned_with_kinematics and show_targets:
                episode_kinematics = kinematics_data[kinematics_data["episode_number"] == episode]
                if not episode_kinematics.empty:
                    # Get a unique target position for this episode
                    tx = episode_kinematics["target_position_x"].iloc[0]
                    ty = episode_kinematics["target_position_y"].iloc[0]
                    tz = episode_kinematics["target_position_z"].iloc[0]

                    # Check if values are arrays and extract if needed
                    if isinstance(tx, (list, np.ndarray)):
                        tx = float(tx[0]) if len(tx) > 0 else np.nan
                    if isinstance(ty, (list, np.ndarray)):
                        ty = float(ty[0]) if len(ty) > 0 else np.nan
                    if isinstance(tz, (list, np.ndarray)):
                        tz = float(tz[0]) if len(tz) > 0 else np.nan

                    # Plot the target
                    ax.scatter(tx, ty, tz, color=colors[i], s=150, marker="*", alpha=0.8)

        # Add a colorbar to show progression of time
        cbar = plt.colorbar(points, ax=ax, pad=0.1)
        cbar.set_label("Timestep")

        # Set labels and title
        ax.set_xlabel(f"Intention Dimension 1 ({x_col})", fontsize=12)
        ax.set_ylabel(f"Intention Dimension 2 ({y_col})", fontsize=12)
        ax.set_zlabel(f"Intention Dimension 3 ({z_col})", fontsize=12)
        ax.set_title("3D Intentions Trajectory", fontsize=16)

        # Add legend (only for a reasonable number of episodes)
        if len(episodes) <= 10:
            ax.legend(loc="upper right")

        # Set equal aspect ratio for better visualization
        max_range = (
            np.array(
                [x_values.max() - x_values.min(), y_values.max() - y_values.min(), z_values.max() - z_values.min()]
            ).max()
            / 2.0
        )

        mid_x = (x_values.max() + x_values.min()) / 2
        mid_y = (y_values.max() + y_values.min()) / 2
        mid_z = (z_values.max() + z_values.min()) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()

        # Add a 2D projection view showing all dimensions as separate subplots
        if len(neuron_columns) > 3:
            n_dims = min(6, len(neuron_columns))  # Show at most 6 dimensions
            fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2 * n_dims), sharex=True)
            if n_dims == 1:
                axes = [axes]  # Make sure axes is always indexable

            for i, episode in enumerate(episodes):
                # Get intentions data for this episode
                if has_episode_col:
                    episode_intentions = intentions_df[intentions_df["episode"] == episode]
                else:
                    episode_intentions = intentions_df

                # Get timesteps from kinematics if available
                if aligned_with_kinematics and kinematics_data is not None:
                    episode_kinematics = kinematics_data[kinematics_data["episode_number"] == episode]
                    if len(episode_kinematics) > 0 and len(episode_kinematics) == len(episode_intentions):
                        # Use the actual timesteps from kinematics
                        x_values = episode_kinematics["index"].values
                    else:
                        x_values = np.arange(len(episode_intentions))
                else:
                    x_values = np.arange(len(episode_intentions))

                # Plot each dimension
                for dim_idx, neuron_col in enumerate(neuron_columns[:n_dims]):
                    ax = axes[dim_idx]

                    ax.plot(
                        x_values,
                        episode_intentions[neuron_col].values,
                        color=colors[i],
                        alpha=0.8,
                        label=f"Episode {episode}" if has_episode_col else f"Episode {i+1}" if dim_idx == 0 else "",
                    )

                    ax.set_ylabel(f"Dim {dim_idx+1}")
                    ax.grid(True, alpha=0.3)

            # Add legend to the first subplot only
            axes[0].legend(loc="best")

            plt.xlabel("Timestep", fontsize=12)
            plt.suptitle("Intention Dimensions Over Time", fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()


if __name__ == "__main__":
    # When run as a script, find and visualize the latest activations file
    plot_intentions_3d()
