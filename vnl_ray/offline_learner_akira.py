import ray
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy,
    NodeAffinitySchedulingStrategy,
)
from ray.util.placement_group import placement_group
import logging
import os
import pandas as pd
import tables
from tqdm import tqdm
from datetime import datetime
import tensorflow_probability as tfp


os.environ["RAY_memory_usage_threshold"] = "1"

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


import time
import os
import dataclasses
import uuid
import hydra
import functools
from omegaconf import DictConfig, OmegaConf
import numpy as np
from acme import specs
from acme import wrappers
from acme.tf import utils as tf2_utils
import sonnet as snt
import tensorflow as tf
from dm_control import composer
import imageio

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
from vnl_ray.utils import plot_reward
from collections import defaultdict


def render_with_rewards(env, policy, observation_spec, episode_num, actuator_type, rollout_length=150, render=False):
    """
    Renders the environment with the rewards progression graph alongside the rendering frames.

    Args:
        env (composer.Environment): The environment to simulate.
        policy (snt.Module): The policy network used to compute actions.
        observation_spec (specs.Array): The observation spec from the environment.
        rollout_length (int, optional): Number of steps to render in each rollout. Defaults to 150.

    Returns:
        tuple: A tuple containing:
            - concat_frames (list of np.ndarray): Rendered frames with reward plot concatenated.
            - activation_collection (list): Collected activations for each timestep.
            - kinematics_collection (list): Collected kinematic data for each timestep.
    """
    frames, reset_idx, reward_channels, activation_collection, kinematics_collection = render_with_rewards_info(
        env, policy, observation_spec, episode_num, actuator_type, rollout_length=rollout_length, render=render
    )
    rewards = defaultdict(list)
    reward_keys = env.task._reward_keys
    for key in reward_keys:
        rewards[key] += [rcs[key] for rcs in reward_channels]
    concat_frames = []
    episode_start = 0
    # implement reset logics of the reward graph too.
    if render == True:
        for idx, frame in enumerate(frames):
            if len(reset_idx) != 0 and idx == reset_idx[0]:
                reward_plot = plot_reward(idx, episode_start, rewards, terminated=True)
                for _ in range(50):
                    concat_frames.append(np.hstack([frame, reward_plot]))  # create stoppage when episode terminates
                reset_idx.pop(0)
                episode_start = idx
                continue
            concat_frames.append(np.hstack([frame, plot_reward(idx, episode_start, rewards)]))
    return concat_frames, activation_collection, kinematics_collection


def render_with_rewards_info(
    env,
    policy,
    observation_spec,
    episode_num,
    actuator_type,
    rollout_length=150,
    render_vision_if_available=False,
    render=False,
):
    """
    Generates a rollout with reward-related information and collects kinematic data for geoms, bodies, joint angles, velocities, and accelerations.

    Args:
        env (composer.Environment): The environment to simulate.
        policy (snt.Module): The policy network used to compute actions.
        observation_spec (specs.Array): The observation spec from the environment.
        rollout_length (int, optional): Number of steps to render in each rollout. Defaults to 150.
        render_vision_if_available (bool, optional): Whether to render vision-based output. Defaults to True.

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

    prev_geom_positions = None  # To compute velocity as finite difference
    prev_body_positions = None  # To compute velocity for bodies
    prev_finger_positions = None

    render_kwargs = {"width": 640, "height": 480}

    # Get all geom names
    geom_names = [env.physics.model.id2name(i, "geom") for i in range(env.physics.model.ngeom)]

    # Get all body names
    body_names = [env.physics.model.id2name(i, "body") for i in range(env.physics.model.nbody)]

    # Get joint names
    joint_names = [env.physics.model.id2name(i, "joint") for i in range(env.physics.model.njnt)]

    for i in range(rollout_length):
        if render == True:
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
                print(f"Warning: Geom '{geom_name}' not found.")

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
                print(f"Warning: Body '{body_name}' not found.")

        # Collect joint angles
        joint_angles = {}
        for joint_name in joint_names:
            try:
                joint_angles[joint_name] = env.physics.named.data.qpos[joint_name].copy()
            except KeyError:
                print(f"Warning: Joint '{joint_name}' not found.")

        # Calculate additional pose and trial information
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
                "pose_and_trial_info": {
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
                },
            }
        )

        # Update previous positions for velocity and acceleration calculation
        prev_geom_positions = geom_positions
        prev_body_positions = body_positions
        prev_finger_positions = {"mouse/finger_tip": finger_pos}

        inputs = tf2_utils.add_batch_dim(timestep.observation)
        inputs = tf2_utils.batch_concat(inputs)

        policy_output = policy(inputs, return_activations=True)
        if isinstance(policy_output, tuple) and len(policy_output) == 2:
            action, activations = policy_output
        elif isinstance(policy_output, tuple) and len(policy_output) == 3:
            action, intentions, activations = policy_output
        else:
            action = policy_output
            activations = {}

        activation_collection.append(activations)

        # Simply use the deterministic mean action without scaling
        action = action.mean()
        # Optionally, save the raw action for analysis
        action_numpy = action.numpy() if hasattr(action, "numpy") else action
        # Update kinematics to store the action
        if kinematics_collection:
            kinematics_collection[-1]["action"] = action_numpy

        timestep = env.step(action)
        reward_channels.append(env.task.last_reward_channels)
        if timestep.step_type == 2:
            reset_idx.append(i)

    return frames, reset_idx, reward_channels, activation_collection, kinematics_collection


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


def run_simulation(learner, actuator_type, num_episodes, rollout_length, render=False):
    """
    Runs the simulation for a specified number of episodes and collects activation and kinematic data.

    Args:
        learner (Learner): The RL learner.
        actuator_type (str): The type of actuator used in the simulation.

    Returns:
        tuple: A tuple containing:
            - all_activations_per_episode (list): Collected activations for each episode.
            - all_kinematics_per_episode (list): Collected kinematics for each episode.
    """
    # Begin simulation
    num_episodes = num_episodes
    rollout_length = rollout_length
    all_activations_per_episode = []
    all_kinematics_per_episode = []
    video_dir = "/root/vast/eric/vnl-ray/videos/"
    policy = learner._policy_network

    # Run the simulation for the specified number of episodes
    for episode in tqdm(range(num_episodes)):
        print(f"Starting Episode {episode + 1}")

        # Create fresh environment with unique seed for each episode
        env = mouse_reach(actuator_type=actuator_type)
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env, clip=False)

        # Explicitly set a different random seed for this episode
        # env.task.random_state.seed(episode + 42)

        # Get observation spec from this environment instance
        observation_spec = env.observation_spec()

        # Run episode with this environment
        frames, activation_collection, kinematics_collection = render_with_rewards(
            env, policy, observation_spec, episode, actuator_type, rollout_length=rollout_length, render=render
        )

        all_activations_per_episode.append(activation_collection)
        all_kinematics_per_episode.append(kinematics_collection)

        if render:
            # Save the video for each episode
            video_path = f"{video_dir}mouse_reach_episode_{actuator_type}_{episode + 1}_checkpoint.mp4"
            with imageio.get_writer(video_path, fps=1 / env.control_timestep()) as video:
                for frame in frames:
                    video.append_data(frame)
            print(f"Episode {episode + 1} finished and saved at {video_path}")

    return all_activations_per_episode, all_kinematics_per_episode


def instantiate_learner(config):
    """
    Initializes and returns a distributed learner for reinforcement learning using
    the DMPO (Distributed Maximum a Posteriori Policy Optimization) algorithm.

    This function sets up the environment, network, and learner configurations based on
    the provided configuration dictionary. It also handles creating multiple replay servers,
    network factories, and learner specifications.

    Args:
        config (DictConfig): A configuration object that contains the following keys:

    Returns:
        Learner: A configured instance of the Learner class, which is ready for training.
    """
    from vnl_ray.agents.ray_distributed_dmpo import (
        DMPOConfig,
        ReplayServer,
        Learner,
        EnvironmentLoop,
    )

    PYHTONPATH = os.path.dirname(os.path.dirname(vnl_ray.__file__))
    LD_LIBRARY_PATH = os.environ["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in os.environ else ""

    # Defer specifying CUDA_VISIBLE_DEVICES to sub-processes.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    print("\nRay context:")
    print(ray_context)

    ray_resources = ray.available_resources()
    print("\nAvailable Ray cluster resources:")
    print(ray_resources)

    # Create environment factory RL task.
    # Cannot parametrize it because it failed to serialize functions
    def environment_factory_mouse_reach() -> "composer.Environment":
        env = tasks["mouse_reach"](actuator_type=config.run_config.actuator_type)
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_run_gaps() -> "composer.Environment":
        env = tasks["run-gaps"]()
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_two_taps() -> "composer.Environment":
        env = tasks["two-taps"]()
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_maze_forage() -> "composer.Environment":
        env = tasks["maze-forage"]()
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_bowl_escape() -> "composer.Environment":
        env = tasks["escape-bowl"]()
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_imitation_humanoid() -> "composer.Environment":
        """Creates replicas of environment for the agent."""
        env = tasks["humanoid_imitation"](config["ref_traj_path"])
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_imitation_rodent(
        termination_error_threshold=0.12, always_init_at_clip_start=False
    ) -> "composer.Environment":
        """
        Creates replicas of environment for the agent. random range controls the
        range of the uniformed distributed termination logics
        """
        env = tasks["rodent_imitation"](
            config["ref_traj_path"],
            reward_term_weights=config["reward_term_weights"] if "reward_term_weights" in config else None,
            termination_error_threshold=termination_error_threshold,
            always_init_at_clip_start=always_init_at_clip_start,
        )
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    environment_factories = {
        "run-gaps": environment_factory_run_gaps,
        "maze-forage": environment_factory_maze_forage,
        "escape-bowl": environment_factory_bowl_escape,
        "two-taps": environment_factory_two_taps,
        "general": environment_factory_run_gaps,
        "imitation_humanoid": environment_factory_imitation_humanoid,
        "imitation_rodent": functools.partial(
            environment_factory_imitation_rodent,
            # termination_error_threshold=config["termination_error_threshold"], # TODO modify the config yaml for the imitation learning too.
        ),
        "mouse_reach": environment_factory_mouse_reach,
    }

    # Dummy environment and network for quick use, deleted later. # create this earlier to access the obs
    dummy_env = environment_factories[config.run_config["task_name"]]()

    # Create network factory for RL task. Config specify different ANN structures
    if config.learner_network["use_intention"]:
        network_factory = make_network_factory_dmpo_intention(
            task_obs_size=get_task_obs_size(
                dummy_env.observation_spec(), config.run_config["agent_name"], config.obs_network["visual_feature_size"]
            ),
            encoder_layer_sizes=config.learner_network["encoder_layer_sizes"],
            decoder_layer_sizes=config.learner_network["decoder_layer_sizes"],
            critic_layer_sizes=config.learner_network["critic_layer_sizes"],
            intention_size=config.learner_network["intention_size"],
            use_tfd_independent=True,  # for easier KL calculation
            use_visual_network=config.obs_network["use_visual_network"],
            visual_feature_size=config.obs_network["visual_feature_size"],
            mid_layer_sizes=(
                config.learner_network["mid_layer_sizes"] if config.learner_network["use_multi_decoder"] else None
            ),
            high_level_intention_size=(
                config.learner_network["high_level_intention_size"]
                if config.learner_network["use_multi_decoder"]
                else None
            ),
        )
    else:
        # online settings
        network_factory = make_network_factory_dmpo(
            action_spec=dummy_env.action_spec(),
            policy_layer_sizes=config.learner_network["policy_layer_sizes"],
            critic_layer_sizes=config.learner_network["critic_layer_sizes"],
        )

    dummy_net = network_factory(dummy_env.action_spec())  # we should share this net for joint training
    # Get full environment specs.
    environment_spec = specs.make_environment_spec(dummy_env)

    # This callable will be calculating penalization cost by converting canonical
    # actions to real (not wrapped) environment actions inside DMPO agent.
    penalization_cost = None  # PenalizationCostRealActions(dummy_env.environment.action_spec())
    # Distributed DMPO agent configuration.
    dmpo_config = DMPOConfig(
        num_actors=config.env_params["num_actors"],
        batch_size=config.learner_params["batch_size"],
        discount=config.learner_params["discount"],
        prefetch_size=1024,  # aggresive prefetch param, because we have large amount of data
        num_learner_steps=1000,
        min_replay_size=50_000,
        max_replay_size=4_000_000,
        samples_per_insert=None,  # allow less sample per insert to allow more data in # None is only min limiter
        n_step=50,
        num_samples=20,
        policy_loss_module=policy_loss_module_dmpo(
            epsilon=0.1,
            epsilon_mean=0.0025,
            epsilon_stddev=1e-7,
            action_penalization=True,
            epsilon_penalty=0.1,
            penalization_cost=penalization_cost,
        ),
        policy_optimizer=snt.optimizers.Adam(config.learner_params["policy_optimizer_lr"]),  # reduce the lr
        critic_optimizer=snt.optimizers.Adam(config.learner_params["critic_optimizer_lr"]),
        dual_optimizer=snt.optimizers.Adam(config.learner_params["dual_optimizer_lr"]),
        target_critic_update_period=107,
        target_policy_update_period=101,
        actor_update_period=5_000,
        log_every=30,
        logger=make_default_logger,
        logger_save_csv_data=False,
        checkpoint_max_to_keep=None,
        checkpoint_directory=f"./training/ray-{config.run_config['agent_name']}-{config.run_config['task_name']}-ckpts/",
        checkpoint_to_load=config.learner_params["checkpoint_to_load"],
        print_fn=None,  # print # this causes issue pprint does not work
        userdata=dict(),
        kickstart_teacher_cps_path=(
            config.learner_params["kickstart_teacher_cps_path"]
            if "kickstart_teacher_cps_path" in config.learner_params
            else None
        ),  # specify the location of the kickstarter teacher policy's cps
        kickstart_epsilon=(
            config.learner_params["kickstart_epsilon"] if "kickstart_epsilon" in config.learner_params else 0
        ),
        time_delta_minutes=30,
        eval_average_over=config.eval_params["eval_average_over"],
        KL_weights=(0, 0),
        # specify the KL with intention & action output layer # do not penalize the output layer # disabled it for now.
        load_decoder_only=(
            config.learner_params["load_decoder_only"] if "load_decoder_only" in config.learner_params else False
        ),
        froze_decoder=config.learner_params["froze_decoder"] if "froze_decoder" in config.learner_params else False,
    )

    dmpo_dict_config = dataclasses.asdict(dmpo_config)
    merged_config = dmpo_dict_config | OmegaConf.to_container(config)  # merged two config

    logger_kwargs = {"config": merged_config}
    dmpo_config.userdata["logger_kwargs"] = logger_kwargs

    # Print full job config and full environment specs.
    print("\n", dmpo_config)
    print("\n", dummy_net)
    print("\nobservation_spec:\n", dummy_env.observation_spec())
    print("\naction_spec:\n", dummy_env.action_spec())
    print("\ndiscount_spec:\n", dummy_env.discount_spec())
    print("\nreward_spec:\n", dummy_env.reward_spec(), "\n")

    # Environment variables for learner, actor, and replay buffer processes.
    runtime_env_learner = {
        "env_vars": {
            "MUJOCO_GL": "osmesa",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "PYTHONPATH": PYHTONPATH,
            "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
        }
    }
    runtime_env_actor = {
        "env_vars": {
            "MUJOCO_GL": "osmesa",
            "CUDA_VISIBLE_DEVICES": "-1",  # CPU-actors don't use CUDA.
            "PYTHONPATH": PYHTONPATH,
            "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
        }
    }

    # Define resources and placement group
    gpu_pg = placement_group([{"GPU": 1, "CPU": 10}], strategy="STRICT_PACK")

    # === Create Replay Server.
    runtime_env_replay = {
        "env_vars": {
            "PYTHONPATH": PYHTONPATH,  # Also used for counter.
        }
    }

    ReplayServer = ray.remote(
        num_gpus=0,
        runtime_env=runtime_env_replay,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=gpu_pg),  # test out performance w/o
    )(ReplayServer)

    replay_servers = dict()  # {task_name: addr} # TODO: Probably could simplify this logic quite a bit
    servers = []
    if "actors_envs" in config:
        dmpo_config.max_replay_size = dmpo_config.max_replay_size // 4  # reduce each replay buffer size by 4.
        for name, num_actors in config.actors_envs.items():
            if num_actors != 0:
                if not config["separate_replay_servers"]:
                    # mixed experience replay buffers
                    replay_server = ReplayServer.remote(
                        dmpo_config, environment_spec
                    )  # each envs will share the same environment spec and dmpo_config
                    addr = ray.get(replay_server.get_server_address.remote())
                    replay_server = RemoteAsLocal(replay_server)
                    servers.append(replay_server)
                    replay_servers["general"] = addr
                    print("SINGLE: Started Single replay server for this task.")
                    break
                elif "num_replay_servers" in config and config["num_replay_servers"] != 0:
                    dmpo_config.max_replay_size = (
                        dmpo_config.max_replay_size // config["num_replay_servers"]
                    )  # shrink down the replay size correspondingly
                    for i in range(config["num_replay_servers"]):
                        _name = f"{name}-{i+1}"
                        # multiple replay server for load balancing
                        replay_server = ReplayServer.remote(
                            dmpo_config, environment_spec
                        )  # each envs will share the same environment spec and dmpo_config
                        addr = ray.get(replay_server.get_server_address.remote())
                        print(f"MULTIPLE: Started Replay Server for task {_name} on {addr}")
                        replay_servers[_name] = addr
                        replay_server = RemoteAsLocal(replay_server)
                        # this line is essential to keep a refernce to the replay server
                        # otherwise the object will be garbage collected and clean out
                        servers.append(replay_server)
                        time.sleep(0.1)
                        # multiple replay server setup
                else:
                    replay_server = ReplayServer.remote(
                        dmpo_config, environment_spec
                    )  # each envs will share the same environment spec and dmpo_config
                    addr = ray.get(replay_server.get_server_address.remote())
                    print(f"MULTIPLE: Started Replay Server for task {name} on {addr}")
                    replay_servers[name] = addr
                    replay_server = RemoteAsLocal(replay_server)
                    # this line is essential to keep a refernce to the replay server
                    # otherwise the object will be garbage collected and clean out
                    servers.append(replay_server)
                    time.sleep(0.1)
    else:
        if "num_replay_servers" in config.env_params and config.env_params["num_replay_servers"] != 0:
            dmpo_config.max_replay_size = (
                dmpo_config.max_replay_size // config.env_params["num_replay_servers"]
            )  # shrink down the replay size correspondingly
            for i in range(config.env_params["num_replay_servers"]):
                name = f"{config.run_config['task_name']}-{i+1}"
                # multiple replay server for load balancing
                replay_server = ReplayServer.remote(
                    dmpo_config, environment_spec
                )  # each envs will share the same environment spec and dmpo_config
                addr = ray.get(replay_server.get_server_address.remote())
                print(f"MULTIPLE: Started Replay Server for task {name} on {addr}")
                replay_servers[name] = addr
                replay_server = RemoteAsLocal(replay_server)
                # this line is essential to keep a refernce to the replay server
                # otherwise the object will be garbage collected and clean out
                servers.append(replay_server)
                time.sleep(0.5)
        else:
            # single replay server
            replay_server = ReplayServer.remote(dmpo_config, environment_spec)
            addr = ray.get(replay_server.get_server_address.remote())
            print(f"Started Replay Server on {addr}")
            replay_servers[config.run_config["task_name"]] = addr

    # === Create Counter.
    counter = ray.remote(PicklableCounter)  # This is class (direct call to ray.remote decorator).
    counter = counter.remote()  # Instantiate.
    counter = RemoteAsLocal(counter)

    # === Create Learner.

    learner = Learner(
        replay_servers,
        counter,
        environment_spec,
        dmpo_config,
        network_factory,
    )

    # Simplified checkpoint handling - remove the expect_partial() logic
    if hasattr(learner, "_checkpointer") and hasattr(learner._checkpointer, "_checkpoint"):
        if learner._checkpointer._checkpoint is not None:
            print(f"Checkpoint loaded from: {config.learner_params['checkpoint_to_load']}")
            print("Note: You may see warnings about unrestored variables - these are expected and can be ignored.")

    # The Learner class already handles loading the checkpoint from config.learner_params["checkpoint_to_load"]
    # No need to manually load checkpoints - the specified checkpoint in the config is being used

    # Let's verify the policy network type for debugging purposes
    policy = learner._policy_network
    print(f"Policy network type: {type(policy).__name__}")
    print(f"Using checkpoint from: {config.learner_params['checkpoint_to_load']}")

    # Check if decoder is frozen as expected
    if hasattr(policy, "decoder") and hasattr(policy.decoder, "_trainable_variables"):
        trainable_count = sum(1 for v in policy.decoder._trainable_variables if v._trainable)
        print(f"Decoder has {trainable_count} trainable variables out of {len(policy.decoder._trainable_variables)}")

    return learner


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="train_config_mouse_reach_offline_akira",
)
def main(config: DictConfig) -> None:
    print("CONFIG:", config)
    # Force actuator_type to use the torque model
    actuator_type = config.run_config.actuator_type

    # Create a learner that loads the checkpoint but doesn't train
    config.learner_params["froze_decoder"] = True  # Ensure decoder is frozen

    # Instantiate learner with the frozen checkpoint
    learner = instantiate_learner(config)

    # Print network structure to verify
    policy = learner._policy_network
    print(f"Policy network structure: {type(policy).__name__}")

    # Verify trainable status for debugging
    for var in policy.trainable_variables:
        if "decoder" in var.name:
            print(f"Decoder variable {var.name} trainable: {var._trainable}")

    # Run simulation to collect activations and kinematics
    print(f"Running simulation with actuator type: {actuator_type}")
    all_activations_per_episode, all_kinematics_per_episode = run_simulation(
        learner,
        actuator_type,
        num_episodes=config.get("num_episodes", 5),  # Default or from config
        rollout_length=config.env_params.get("rollout_length", 100),  # Read from env_params with default of 100
        render=config.get("render", True),  # Default or from config
    )

    # Get current datetime string for unique saving
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use absolute path to avoid path duplication issues with Hydra
    base_dir = os.path.abspath("/root/vast/eric/vnl-ray")

    # Save paths with actuator type and timestamp
    activations_h5_file_path = os.path.join(
        base_dir, "training", "activations", f"activations_{actuator_type}_{datetime_str}.h5"
    )
    kinematics_h5_file_path = os.path.join(
        base_dir, "training", "kinematics", f"kinematics_{actuator_type}_{datetime_str}.h5"
    )

    # Print the actual paths being used
    print(f"Saving activations to: {activations_h5_file_path}")
    print(f"Saving kinematics to: {kinematics_h5_file_path}")

    # Process and save using the generalized function
    process_and_save_activation_collection(all_activations_per_episode, activations_h5_file_path)
    process_and_save_kinematics_collection(all_kinematics_per_episode, kinematics_h5_file_path)

    print("Analysis complete!")

    # Print full paths again to help with debugging/finding files
    print(f"Activations saved to: {os.path.abspath(activations_h5_file_path)}")
    print(f"Kinematics saved to: {os.path.abspath(kinematics_h5_file_path)}")


if __name__ == "__main__":
    main()
