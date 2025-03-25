"""
Script for offline evaluation of trained DMPO agents with Ray.

This script loads a trained model checkpoint, runs evaluation episodes,
and records kinematics and activation information for analysis.
"""

import ray
import os
import logging
import hydra
import functools
import numpy as np
import time
import dataclasses
from omegaconf import DictConfig, OmegaConf
from acme import specs
from acme import wrappers
import tensorflow as tf
import sonnet as snt
from dm_control import composer
from acme.tf import utils as tf2_utils
import pandas as pd  # Add pandas for HDF5 storage

os.environ["RAY_memory_usage_threshold"] = "1"

import vnl_ray
from vnl_ray.agents.remote_as_local_wrapper import RemoteAsLocal
from vnl_ray.agents.counting import PicklableCounter
from vnl_ray.agents.network_factory import policy_loss_module_dmpo
from vnl_ray.agents.losses_mpo import PenalizationCostRealActions
from vnl_ray.tasks.basic_rodent_2020 import (
    rodent_run_gaps,
    rodent_maze_forage,
    rodent_escape_bowl,
    rodent_two_touch,
    walk_humanoid,
    rodent_walk_imitation,
)
from vnl_ray.tasks.mouse_reach import mouse_reach
from vnl_ray.fly_envs import (
    walk_on_ball,
    vision_guided_flight,
    walk_imitation as fly_walk_imitation,
)
from vnl_ray.default_logger import make_default_logger
from vnl_ray.single_precision import SinglePrecisionWrapper
from vnl_ray.agents.network_factory import make_network_factory_dmpo
from vnl_ray.agents.intention_network_factory import (
    make_network_factory_dmpo as make_network_factory_dmpo_intention,
)
from vnl_ray.tasks.task_utils import get_task_obs_size

# Remove any pre-existing LD_LIBRARY_PATH and override it
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ["LD_LIBRARY_PATH"] = "/root/miniforge3/envs/flybody/lib"

PYTHONPATH = os.path.dirname(os.path.dirname(vnl_ray.__file__))

# Initialize Ray with minimal configuration
try:
    # Try connecting to existing Ray cluster.
    ray_context = ray.init(
        address="auto",
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        logging_level=logging.INFO,
    )
except:
    # Spin up new Ray cluster.
    ray_context = ray.init(include_dashboard=True, dashboard_host="0.0.0.0", logging_level=logging.INFO)


# Task definitions
tasks = {
    "run-gaps": rodent_run_gaps,
    "maze-forage": rodent_maze_forage,
    "escape-bowl": rodent_escape_bowl,
    "two-taps": rodent_two_touch,
    "rodent_imitation": rodent_walk_imitation,
    "fly_imitation": fly_walk_imitation,
    "humanoid_imitation": walk_humanoid,
    "mouse_reach": mouse_reach,
}


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="train_config_mouse_reach_offline_akira",
)
def main(config: DictConfig) -> None:
    print("CONFIG:", config)

    from vnl_ray.agents.ray_distributed_dmpo import DMPOConfig

    print("\nRay context:")
    print(ray.cluster_resources())

    # Create environment factory for the specified task
    def environment_factory():
        env = tasks[config.run_config.task_name](
            actuator_type=config.run_config.actuator_type if hasattr(config.run_config, "actuator_type") else None,
            config=config,
        )
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    # Create environment and get specs
    env = environment_factory()
    environment_spec = specs.make_environment_spec(env)

    # Create network factory
    network_factory = make_network_factory_dmpo_intention(
        task_obs_size=get_task_obs_size(
            env.observation_spec(), config.run_config["agent_name"], config.obs_network["visual_feature_size"]
        ),
        encoder_layer_sizes=config.learner_network["encoder_layer_sizes"],
        decoder_layer_sizes=config.learner_network["decoder_layer_sizes"],
        critic_layer_sizes=config.learner_network["critic_layer_sizes"],
        intention_size=config.learner_network["intention_size"],
        use_tfd_independent=True,
        use_visual_network=config.obs_network["use_visual_network"],
        visual_feature_size=config.obs_network["visual_feature_size"],
        mid_layer_sizes=config.learner_network.get("mid_layer_sizes", None),
        high_level_intention_size=config.learner_network.get("high_level_intention_size", None),
    )

    # Create network
    network = network_factory(env.action_spec())

    # Extract the policy object - careful with the dictionary structure
    try:
        print(f"Network keys: {list(network.keys())}")

        # Extract the intention network directly
        if "policy" in network:
            policy = network["policy"]
            print(f"Found policy: {policy}")
        else:
            raise ValueError(f"Policy key not found in network dict with keys: {list(network.keys())}")

        # Create a simple dummy observation for initialization
        observation_spec = env.observation_spec()
        dummy_obs = {}
        for key, spec in observation_spec.items():
            dummy_obs[key] = np.zeros(spec.shape, dtype=spec.dtype)

        # Add batch dimension and concatenate
        batched_obs = tf2_utils.add_batch_dim(dummy_obs)
        dummy_input = tf2_utils.batch_concat(batched_obs)

        # Call the policy directly
        try:
            print(f"Calling policy with input shape: {dummy_input.shape}")
            policy_output = policy(dummy_input)
            print(f"Policy output: {policy_output}")

            # Now we can access trainable variables
            trainable_vars = policy.trainable_variables
            print(f"Found {len(trainable_vars)} trainable variables")

            # Save initial weights for comparison
            initial_weights = {var.name: var.numpy().copy() for var in trainable_vars}
        except Exception as e:
            print(f"Error calling policy: {e}")
            print(f"Policy type: {type(policy)}")
            raise e
    except Exception as e:
        print(f"ERROR with network structure: {e}")
        print(f"Network type: {type(network)}")
        raise e

    snapshot_path = config["snapshot_path"]
    print(f"Loading snapshot from saved model: {snapshot_path}")
    try:
        loaded_model = tf.saved_model.load(snapshot_path)

        # Get variables from both models and organize them
        network_vars = {var.name: var for var in policy.trainable_variables}
        loaded_vars = {v.name: v for v in loaded_model.variables}

        # Sort variables by name for clearer comparison
        print("\nNetwork initialized variables:")
        for name in sorted(network_vars.keys()):
            var = network_vars[name]
            print(f"{name}: shape={var.shape}, dtype={var.dtype}")

        print("\nSnapshot model variables:")
        for name in sorted(loaded_vars.keys()):
            var = loaded_vars[name]
            print(f"{name}: shape={var.shape}, dtype={var.dtype}")

        # Attempt intelligent variable matching
        print("\nAttempting variable transfer...")
        transfer_stats = {"exact_match": 0, "shape_mismatch": 0, "not_found": 0}
        transferred_vars = []

        for net_name, net_var in network_vars.items():
            # Try exact name match first
            if net_name in loaded_vars:
                loaded_var = loaded_vars[net_name]
                if net_var.shape == loaded_var.shape:
                    net_var.assign(loaded_var)
                    print(f"✓ Exact match: {net_name}")
                    transfer_stats["exact_match"] += 1
                    transferred_vars.append(net_name)
                else:
                    print(f"✗ Shape mismatch for {net_name}: network={net_var.shape}, loaded={loaded_var.shape}")
                    transfer_stats["shape_mismatch"] += 1
            else:
                # Try to find alternative matches (removing leading/trailing spaces, etc.)
                cleaned_name = net_name.strip()
                alternative_matches = [k for k in loaded_vars.keys() if cleaned_name in k or k in cleaned_name]

                if alternative_matches:
                    print(f"? No exact match for {net_name}, but found alternatives: {alternative_matches}")
                    # Try each alternative
                    for alt_name in alternative_matches:
                        loaded_var = loaded_vars[alt_name]
                        if net_var.shape == loaded_var.shape:
                            net_var.assign(loaded_var)
                            print(f"✓ Alternative match: {net_name} ← {alt_name}")
                            transfer_stats["exact_match"] += 1
                            transferred_vars.append(net_name)
                            break
                    else:
                        print(f"✗ No compatible alternative found for {net_name}")
                        transfer_stats["not_found"] += 1
                else:
                    print(f"✗ No match found for {net_name}")
                    transfer_stats["not_found"] += 1

        # Summary
        print("\nTransfer summary:")
        print(f"  - Variables found with exact shape match: {transfer_stats['exact_match']}")
        print(f"  - Variables with shape mismatch: {transfer_stats['shape_mismatch']}")
        print(f"  - Variables with no match found: {transfer_stats['not_found']}")
        print(f"  - Total transfer success rate: {transfer_stats['exact_match']}/{len(network_vars)} variables")

        if transfer_stats["exact_match"] > 0:
            print("\nPartial model loading successful")
        else:
            print("\nWARNING: No variables were transferred!")

    except Exception as e:
        print(f"WARNING: Failed to load snapshot! Error: {e}")
        return

    # Compute and print layer-wise L2 norm differences to verify transfer.
    print("\nLayer-wise weight differences summary (L2 norm):")
    seen_vars = set()  # Track which variables we've already reported
    unchanged_vars = []  # Track variables with no change

    for var in policy.trainable_variables:
        # Skip duplicates
        if var.name in seen_vars:
            continue
        seen_vars.add(var.name)

        restored = var.numpy()
        if var.name in initial_weights:
            init = initial_weights[var.name]
            if restored.shape == init.shape:
                diff_norm = np.linalg.norm(restored - init)
                # Check if change is significant
                if diff_norm < 1e-6:
                    status = "❌ (unchanged)"
                    unchanged_vars.append(var.name)
                else:
                    status = "✅"
                print(f"{var.name}: L2 diff = {diff_norm:.4e} {status}")
            else:
                print(f"Warning: {var.name} shape mismatch: restored {restored.shape} vs initial {init.shape}")
        else:
            print(f"Warning: {var.name} not found in initial weights")

    # Detailed examination of unchanged variables
    if unchanged_vars:
        print("\n🔍 Detailed examination of unchanged variables:")
        for var_name in unchanged_vars:
            var = next(v for v in policy.trainable_variables if v.name == var_name)
            var_data = var.numpy()

            print(f"\n  Variable: {var_name}")
            print(f"    Shape: {var_data.shape}")

            # Check if the bias is all zeros (common initialization for bias)
            all_zeros = np.all(np.abs(var_data) < 1e-6)
            print(f"    All zeros: {all_zeros}")

            # Print a sample of values
            if var_data.size <= 10:
                print(f"    Values: {var_data.flatten()}")
            else:
                print(f"    First 5 values: {var_data.flatten()[:5]}")
                print(f"    Min/Max/Mean: {var_data.min():.6f}/{var_data.max():.6f}/{var_data.mean():.6f}")

            # Check if this variable was among those we found in the snapshot
            if var_name in transferred_vars:
                print(f"    ✅ Was listed among transferred variables")
                if var_name in loaded_vars:
                    snapshot_data = loaded_vars[var_name].numpy()
                    identical = np.allclose(var_data, snapshot_data, atol=1e-6)
                    print(f"    📊 Snapshot values identical to current: {identical}")

                    # Compare snapshot with initial values
                    if var_name in initial_weights:
                        init_data = initial_weights[var_name]
                        identical_to_init = np.allclose(snapshot_data, init_data, atol=1e-6)
                        print(f"    📊 Snapshot values identical to initial: {identical_to_init}")

                        # If all are zeros, that's a common initialization pattern for biases
                        if all_zeros and identical_to_init:
                            print(
                                f"    💡 This is likely a bias variable initialized to zeros and not changed in training"
                            )
            else:
                print(f"    ❌ Not listed among transferred variables")

    # Setup recording
    num_eval_episodes = 100  # Number of evaluation episodes to run
    output_dir = os.path.join(
        os.path.dirname(config["snapshot_path"]), "offline_eval", os.path.basename(config["snapshot_path"])
    )
    os.makedirs(output_dir, exist_ok=True)
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    print(f"Will run {num_eval_episodes} evaluation episodes and save results to {output_dir}")

    # Create data structures to store results
    all_rewards = []
    all_observations = []
    all_actions = []
    all_activations = []  # Store network activations
    all_kinematics = []  # Store kinematics data

    # Add these configuration parameters with defaults
    action_scale = config.get("action_scale", 1.0)  # Default to 3x stronger actions
    use_stochastic = config.get("use_stochastic", False)  # Default to deterministic

    # print(f"\nAction parameters:")
    # print(f"  - Action scale factor: {action_scale}x (multiplies action magnitude)")
    # print(f"  - Action selection: {'stochastic sampling' if use_stochastic else 'deterministic mean'}")

    # Run evaluation episodes
    for episode in range(num_eval_episodes):
        print(f"Running evaluation episode {episode+1}/{num_eval_episodes}")

        # Create new environment instance for each episode to ensure fresh state
        eval_env = environment_factory()

        if config.get("render", True):
            # Use the comprehensive rendering function with action scaling
            episode_rewards, episode_observations, episode_actions, episode_activations, episode_kinematics = (
                render_episode(
                    eval_env, policy, episode + 1, video_dir, use_stochastic=use_stochastic, action_scale=action_scale
                )
            )

            # Store episode data
            all_rewards.append(episode_rewards["distance_reward"] if "distance_reward" in episode_rewards else [])
            all_observations.append(episode_observations)
            all_actions.append(episode_actions)
            all_activations.append(episode_activations)
            all_kinematics.append(episode_kinematics)

            print(
                f"Episode {episode+1} complete. Total reward: {sum(episode_rewards['distance_reward']) if 'distance_reward' in episode_rewards else 0}"
            )
            print(f"Sample action: {episode_actions[0] if episode_actions else 'No actions recorded'}")
        else:
            # Standard non-rendering evaluation
            timestep = eval_env.reset()
            episode_rewards = []
            episode_observations = []
            episode_actions = []

            # Run episode until termination
            while not timestep.last():
                # Get observation
                observation = timestep.observation
                episode_observations.append(observation)

                # Preprocess observation using the same utility as in training
                inputs = tf2_utils.add_batch_dim(observation)  # Add batch dimension
                inputs = tf2_utils.batch_concat(inputs)  # Concatenate into a single tensor

                # Get action from the policy network
                policy_output = policy(inputs)
                if config.get("use_stochastic", False):
                    action = policy_output.sample().numpy()  # Use stochastic sampling
                else:
                    action = policy_output.mean().numpy()  # Use deterministic mean action

                # Scale action to match environment's action space
                # action = np.clip(action, -1, 1)
                episode_actions.append(action)

                # Step environment
                timestep = eval_env.step(action)
                episode_rewards.append(timestep.reward)

            # Store episode data
            all_rewards.append(episode_rewards)
            all_observations.append(episode_observations)
            all_actions.append(episode_actions)

            print(f"Episode {episode+1} complete. Total reward: {sum(episode_rewards)}")
            print(f"Sample action: {episode_actions[0] if episode_actions else 'No actions recorded'}")

    # Save evaluation results including activations and kinematics
    np.save(os.path.join(output_dir, "rewards.npy"), np.array(all_rewards, dtype=object))
    np.save(os.path.join(output_dir, "observations.npy"), np.array(all_observations, dtype=object))
    np.save(os.path.join(output_dir, "actions.npy"), np.array(all_actions, dtype=object))

    # Process and save activations in HDF5 format
    activation_h5_path = os.path.join(output_dir, "activations.h5")
    process_and_save_activation_collection(all_activations, activation_h5_path)

    # Process and save kinematics in HDF5 format
    kinematics_h5_path = os.path.join(output_dir, "kinematics.h5")
    process_and_save_kinematics_collection(all_kinematics, kinematics_h5_path)

    print(f"Evaluation complete. Data saved to {output_dir}")
    print(f"Activations saved to {activation_h5_path}")
    print(f"Kinematics saved to {kinematics_h5_path}")


# Add this function before the main() function
def offline_logger(
    label,
    steps_key=None,
    task_instance=0,
    save_data=False,
    time_delta=None,
    asynchronous=False,
    print_fn=None,
    serialize_fn=None,
    steps=None,
    **kwargs,
):
    """Simple logger for offline evaluation that doesn't depend on run_name config."""
    from acme.utils.loggers import base

    # Just return a simple terminal logger
    from acme.utils.loggers.terminal import TerminalLogger

    return TerminalLogger(label=label, print_fn=print_fn)


# Import additional modules needed for rendering
import imageio
import os
from collections import defaultdict


# Add utility function for rendering rewards
def plot_reward(timestep, start_timestep, rewards, terminated=False):
    """Creates a simple reward plot for visualization alongside rendered frames."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot rewards
    for key, values in rewards.items():
        if len(values) > 0:
            ax.plot(range(start_timestep, start_timestep + len(values)), values, label=key)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards over Time" if not terminated else "Episode Complete")
    ax.legend()
    ax.grid(True)

    # Highlight current timestep
    if timestep >= start_timestep:
        ax.axvline(x=timestep, color="r", linestyle="--")

    # Convert plot to image
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return img


# Modify the render_episode function to support action scaling
def render_episode(env, policy, episode_num, output_dir, max_steps=300, use_stochastic=False, action_scale=1.0):
    """
    Renders a complete episode and saves it as a video.

    Args:
        env: The environment to render
        policy: The policy to use for action selection
        episode_num: The episode number (for filename)
        output_dir: Directory where to save the video
        max_steps: Maximum number of steps to run
        use_stochastic: Whether to use stochastic sampling for actions
        action_scale: Scaling factor for actions

    Returns:
        Tuple of (rewards, observations, actions, activations, kinematics)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create video writer
    video_path = os.path.join(output_dir, f"episode_{episode_num}.mp4")

    # Call comprehensive state logging function to collect all data
    frames, reset_idx, reward_channels, activation_collection, kinematics_collection = render_with_rewards_info(
        env=env,
        policy=policy,
        observation_spec=env.observation_spec(),
        episode_num=episode_num,
        actuator_type="torque" if hasattr(env, "actuator_type") else None,
        rollout_length=max_steps,
        render=True,
        action_scale=action_scale,  # Pass action scaling
    )

    # Process rewards
    rewards = defaultdict(list)
    reward_keys = env.task._reward_keys if hasattr(env.task, "_reward_keys") else ["reward"]
    for key in reward_keys:
        for rcd in reward_channels:
            if key in rcd:
                rewards[key].append(rcd[key])

    # Extract observations from kinematics
    observations = []
    actions = []

    # Extract actions from kinematics
    for step_data in kinematics_collection:
        if "action" in step_data:
            actions.append(step_data["action"])

    # Save video if we have frames
    if frames and frames[0] is not None:
        # Add final frames showing episode completion
        final_frame = frames[-1]
        for _ in range(30):  # Show completion for 30 frames
            frames.append(final_frame)

        # Save video
        with imageio.get_writer(video_path, fps=60) as writer:
            for frame in frames:
                writer.append_data(frame)

        print(f"Video saved to {video_path}")

    return rewards, observations, actions, activation_collection, kinematics_collection


# Add these utility functions for comprehensive state logging
def flatten_dict(d, parent_key="", sep="/"):
    # Recursively flattens nested dictionaries.
    import tensorflow_probability as tfp

    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # If value is a tfp distribution, use its mean.
        if isinstance(v, tfp.distributions.Distribution):
            mean_val = v.mean()
            items[new_key] = mean_val.numpy() if hasattr(mean_val, "numpy") else mean_val
        elif isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def process_and_save_activation_collection(activation_collections, h5_file_path):
    """
    Processes activation collections with arbitrary layer names and saves them as DataFrames in HDF5 format.
    Now recursively flattens activations to capture encoder, decoder, and intention values.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(h5_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("Analyzing activation data structure...")
    all_keys = set()
    # Use the flattened structure
    for episode_collection in activation_collections:
        for timestep_data in episode_collection:
            flat_data = flatten_dict(timestep_data)
            all_keys.update(flat_data.keys())
    print(f"Found {len(all_keys)} flattened keys: {all_keys}")

    # Prepare container
    final_dfs = {}

    for episode_idx, activation_collection in enumerate(activation_collections):
        print(f"Processing episode {episode_idx+1}/{len(activation_collections)}")
        episode_data = {}
        for timestep_idx, activations in enumerate(activation_collection):
            flat_activations = flatten_dict(activations)
            for key, value in flat_activations.items():
                if key not in episode_data:
                    episode_data[key] = []
                episode_data[key].append(value)
        for key, values in episode_data.items():
            try:
                array = np.squeeze(np.array(values))
                if array.ndim > 1:
                    neuron_columns = [f"neuron_{i+1}" for i in range(array.shape[1])]
                    df = pd.DataFrame(array, columns=neuron_columns)
                else:
                    df = pd.DataFrame({key: values})
                if key not in final_dfs:
                    final_dfs[key] = []
                final_dfs[key].append(df)
            except Exception as e:
                print(f"Error processing {key}: {e}")

    print("Combining data across episodes...")
    combined_dfs = {}
    for key, dfs in final_dfs.items():
        if dfs:
            combined_dfs[key] = pd.concat(dfs, ignore_index=True)

    print(f"Saving {len(combined_dfs)} DataFrames to HDF5...")
    with pd.HDFStore(h5_file_path) as store:
        for key, df in combined_dfs.items():
            store_key = key.replace("/", "_")
            store[store_key] = df
            print(f"Saved activation data for '{key}' as '{store_key}'")
    print(f"All activation data saved to {h5_file_path}")


def process_and_save_kinematics_collection(kinematics_collections, h5_file_path):
    """
    Processes the kinematics collections and saves them into separate keys in HDF5 format.
    - Separate keys for geom positions, velocities, accelerations, etc.
    - Rows: Timesteps.
    - Columns: Geom positions, body positions, joint angles, velocities, accelerations, and episode number.

    Parameters:
    - kinematics_collections: List of lists, where each sublist contains the kinematics data for one episode.
    - h5_file_path: Path to the HDF5 file where the data will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(h5_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize dictionaries to store DataFrames for each key
    geoms_positions = []
    geoms_velocities = []
    geoms_accelerations = []

    bodies_positions = []
    bodies_velocities = []
    bodies_accelerations = []
    bodies_orientations = []

    joints_angles = []

    for episode_idx, kinematics_collection in enumerate(kinematics_collections):
        # Iterate over each timestep and extract positions, velocities, accelerations, and joint angles
        for timestep_idx, kinematics in enumerate(kinematics_collection):
            # Create a flat dictionary for each key based on the timestep
            timestep_dict = {"episode": episode_idx + 1, "timestep": timestep_idx}

            # Geoms: Positions, Velocities, Accelerations
            for geom_name, position in kinematics["geoms"]["positions"].items():
                timestep_dict[f"{geom_name}"] = position  # Store [x, y, z]
            geoms_positions.append(timestep_dict)

            for geom_name, velocity in kinematics["geoms"]["velocities"].items():
                timestep_dict[f"{geom_name}"] = velocity  # Store [vx, vy, vz]
            geoms_velocities.append(timestep_dict)

            for geom_name, acceleration in kinematics["geoms"]["accelerations"].items():
                timestep_dict[f"{geom_name}"] = acceleration  # Store [ax, ay, az]
            geoms_accelerations.append(timestep_dict)

            # Bodies: Positions, Velocities, Accelerations, Orientations
            for body_name, position in kinematics["bodies"]["positions"].items():
                timestep_dict[f"{body_name}"] = position  # Store [x, y, z]
            bodies_positions.append(timestep_dict)

            for body_name, velocity in kinematics["bodies"]["velocities"].items():
                timestep_dict[f"{body_name}"] = velocity  # Store [vx, vy, vz]
            bodies_velocities.append(timestep_dict)

            for body_name, acceleration in kinematics["bodies"]["accelerations"].items():
                timestep_dict[f"{body_name}"] = acceleration  # Store [ax, ay, az]
            bodies_accelerations.append(timestep_dict)

            for body_name, orientation in kinematics["bodies"]["orientations"].items():
                timestep_dict[f"{body_name}"] = orientation  # Store [qw, qx, qy, qz]
            bodies_orientations.append(timestep_dict)

            # Joints: Angles
            for joint_name, angle in kinematics["joints"].items():
                timestep_dict[f"{joint_name}"] = angle  # Store single angle value
            joints_angles.append(timestep_dict)

    # Convert to DataFrames
    geoms_positions_df = pd.DataFrame(geoms_positions)
    geoms_velocities_df = pd.DataFrame(geoms_velocities)
    geoms_accelerations_df = pd.DataFrame(geoms_accelerations)

    bodies_positions_df = pd.DataFrame(bodies_positions)
    bodies_velocities_df = pd.DataFrame(bodies_velocities)
    bodies_accelerations_df = pd.DataFrame(bodies_accelerations)
    bodies_orientations_df = pd.DataFrame(bodies_orientations)

    joints_angles_df = pd.DataFrame(joints_angles)

    # Extract 'pose_and_trial_info' from each timestep across all episodes
    pose_and_trial_info_list = [
        timestep["pose_and_trial_info"]
        for episode in kinematics_collections
        for timestep in episode
        if "pose_and_trial_info" in timestep
    ]

    # Convert to DataFrame
    pose_and_trial_info_df = pd.DataFrame(pose_and_trial_info_list)

    # Save the DataFrames to separate keys in HDF5
    with pd.HDFStore(h5_file_path) as store:
        store["/geoms/positions"] = geoms_positions_df
        store["/geoms/velocities"] = geoms_velocities_df
        store["/geoms/accelerations"] = geoms_accelerations_df

        store["/bodies/positions"] = bodies_positions_df
        store["/bodies/velocities"] = bodies_velocities_df
        store["/bodies/accelerations"] = bodies_accelerations_df
        store["/bodies/orientations"] = bodies_orientations_df

        store["/joints/angles"] = joints_angles_df

        store["/pose_and_trial_info"] = pose_and_trial_info_df

    print(f"Kinematics data saved to {h5_file_path}")


# Modify render_with_rewards_info to apply action scaling
def render_with_rewards_info(
    env,
    policy,
    observation_spec,
    episode_num,
    actuator_type,
    rollout_length=150,
    render_vision_if_available=False,
    render=False,
    action_scale=1.0,  # Add action scale parameter
):
    """
    Generates a rollout with reward-related information and collects kinematic data for geoms, bodies, joint angles, velocities, and accelerations.

    Args:
        env (composer.Environment): The environment to simulate.
        policy (snt.Module): The policy network used to compute actions.
        observation_spec (specs.Array): The observation spec from the environment.
        episode_num (int): Episode number for tracking.
        actuator_type (str): Type of actuator being used.
        rollout_length (int, optional): Number of steps to render in each rollout. Defaults to 150.
        render_vision_if_available (bool, optional): Whether to render vision-based output. Defaults to False.
        render (bool, optional): Whether to render frames. Defaults to False.
        action_scale (float, optional): Scaling factor for actions. Defaults to 1.0.

    Returns:
        tuple: A tuple containing:
            - frames (list of np.ndarray): Rendered frames.
            - reset_idx (list of int): Indexes where the episode resets occurred.
            - reward_channels (list of dict): Collected reward values at each timestep.
            - activation_collection (list): Collected activations at each timestep.
            - kinematics_collection (list): Collected kinematics at each timestep.
    """
    reward_channels = []
    frames = []
    reset_idx = []
    timestep = env.reset()
    activation_collection = []
    kinematics_collection = []
    rewards = []

    prev_geom_positions = None  # To compute velocity as finite difference
    prev_body_positions = None  # To compute velocity for bodies
    prev_finger_positions = None

    render_kwargs = {"width": 600, "height": 400}

    # Get all geom names
    geom_names = [env.physics.model.id2name(i, "geom") for i in range(env.physics.model.ngeom)]

    # Get all body names
    body_names = [env.physics.model.id2name(i, "body") for i in range(env.physics.model.nbody)]

    # Get joint names
    joint_names = [env.physics.model.id2name(i, "joint") for i in range(env.physics.model.njnt)]

    for i in range(rollout_length):
        if render:
            pixels = env.physics.render(camera_id=1, **render_kwargs)
            frames.append(pixels)
        else:
            frames.append(None)

        # Collect kinematic data for geoms
        geom_positions = {}
        geom_velocities = {}
        geom_accelerations = {}

        for geom_name in geom_names:
            try:
                geom_positions[geom_name] = env.physics.named.data.geom_xpos[geom_name].copy()

                # Calculate velocity as finite difference between positions
                if prev_geom_positions:
                    geom_velocities[geom_name] = geom_positions[geom_name] - prev_geom_positions[geom_name]
                else:
                    geom_velocities[geom_name] = np.zeros(3)  # No velocity for the first step

                # Acceleration as difference of velocities
                if prev_geom_positions:
                    geom_accelerations[geom_name] = geom_velocities[
                        geom_name
                    ]  # Current velocity as acceleration for first difference
                else:
                    geom_accelerations[geom_name] = np.zeros(3)  # No acceleration for the first step
            except KeyError:
                # Skip if geom not found
                pass

        # Collect kinematic data for bodies (use position difference to estimate velocity)
        body_positions = {}
        body_velocities = {}
        body_accelerations = {}
        body_orientations = {}

        for body_name in body_names:
            try:
                body_positions[body_name] = env.physics.named.data.xpos[body_name].copy()

                # Calculate velocity as finite difference between positions
                if prev_body_positions:
                    body_velocities[body_name] = body_positions[body_name] - prev_body_positions[body_name]
                else:
                    body_velocities[body_name] = np.zeros(3)  # No velocity for the first step

                # Acceleration as difference of velocities
                if prev_body_positions:
                    body_accelerations[body_name] = body_velocities[
                        body_name
                    ]  # Current velocity as acceleration for first difference
                else:
                    body_accelerations[body_name] = np.zeros(3)  # No acceleration for the first step

                # Orientation using quaternions or rotation matrix
                body_orientations[body_name] = env.physics.named.data.xquat[
                    body_name
                ].copy()  # Or use xmat for rotation matrix
            except KeyError:
                # Skip if body not found
                pass

        # Collect joint angles
        joint_angles = {}
        for joint_name in joint_names:
            try:
                joint_angles[joint_name] = env.physics.named.data.qpos[joint_name].copy()
            except KeyError:
                # Skip if joint not found
                pass

        # Calculate additional pose and trial information
        try:
            to_target = timestep.observation["mouse/to_target"]
            target_size = timestep.observation["mouse/target_size"][0]
            reward = timestep.reward
            target_pos = env.physics.named.data.geom_xpos["mouse/target"].copy()
            finger_pos = env.physics.named.data.geom_xpos["mouse/finger_tip"].copy()

            # Calculate velocity vector (finite difference)
            if prev_finger_positions:
                finger_velocity = finger_pos - prev_finger_positions["mouse/finger_tip"]
            else:
                finger_velocity = np.zeros(3)

            pose_and_trial_info = {
                "index": i,
                "to_target_x": to_target[0],
                "to_target_y": to_target[1],
                "to_target_z": to_target[2],
                "episode_number": episode_num,
                "target_size": target_size,
                "reward": reward,
                "target_position_x": target_pos[0],
                "target_position_y": target_pos[1],
                "target_position_z": target_pos[2],
                "finger_tip_x": finger_pos[0],
                "finger_tip_y": finger_pos[1],
                "finger_tip_z": finger_pos[2],
                "velocity_x": finger_velocity[0],
                "velocity_y": finger_velocity[1],
                "velocity_z": finger_velocity[2],
            }
            prev_finger_positions = {"mouse/finger_tip": finger_pos}
        except (KeyError, ValueError) as e:
            # If specific mouse task elements aren't present, create minimal info
            pose_and_trial_info = {
                "index": i,
                "episode_number": episode_num,
                "reward": timestep.reward if hasattr(timestep, "reward") else 0.0,
            }

        # Store positions, velocities, accelerations, orientations, and joint angles for this timestep
        kinematics_collection.append(
            {
                "geoms": {
                    "positions": geom_positions,
                    "velocities": geom_velocities,
                    "accelerations": geom_accelerations,
                },
                "bodies": {
                    "positions": body_positions,
                    "velocities": body_velocities,
                    "accelerations": body_accelerations,
                    "orientations": body_orientations,
                },
                "joints": joint_angles,
                "pose_and_trial_info": pose_and_trial_info,
            }
        )

        # Update previous positions for velocity and acceleration calculation
        prev_geom_positions = geom_positions
        prev_body_positions = body_positions

        # Get policy inputs and potential activations
        inputs = tf2_utils.add_batch_dim(timestep.observation)
        inputs = tf2_utils.batch_concat(inputs)

        # Check if policy has ability to return activations
        has_return_activations = hasattr(policy, "_call_with_activations")

        if has_return_activations:
            # Call with activations flag
            policy_output = policy(inputs, return_activations=True)
            if isinstance(policy_output, tuple) and len(policy_output) == 2:
                action, activations = policy_output
            elif isinstance(policy_output, tuple) and len(policy_output) == 3:
                action, intentions, activations = policy_output
            else:
                action = policy_output
                activations = {}
        else:
            # Regular call
            policy_output = policy(inputs)
            action = policy_output
            activations = {}

        activation_collection.append(activations)

        # Simply use the deterministic mean action without scaling
        if hasattr(action, "mean"):
            action = action.mean()
        # Optionally, save the raw action for analysis
        action_numpy = action.numpy() if hasattr(action, "numpy") else action

        # Apply action scaling - this is the key change for more aggressive movement
        if action_scale != 1.0:
            action_numpy = action_numpy * action_scale
            # Log for debugging
            if i % 30 == 0:  # Every 30 steps
                print(f"Action before scaling: {action_numpy/action_scale}")
                print(f"Action after scaling: {action_numpy}")
                print(f"Action spec min/max: {env.action_spec().minimum}/{env.action_spec().maximum}")

        # Ensure actions stay within bounds after scaling
        action_numpy = np.clip(action_numpy, env.action_spec().minimum, env.action_spec().maximum)

        # Update kinematics to store the scaled action
        if kinematics_collection:
            kinematics_collection[-1]["action"] = action_numpy

        # Use the scaled action
        timestep = env.step(action_numpy)
        if hasattr(env.task, "last_reward_channels"):
            reward_channels.append(env.task.last_reward_channels)
        if timestep.step_type == 2:
            reset_idx.append(i)

    return frames, reset_idx, reward_channels, activation_collection, kinematics_collection


if __name__ == "__main__":
    main()
