"""Classes for DMPO agent distributed with Ray."""

from typing import Iterator, Callable, List
import socket
import dataclasses
import copy
import logging
import os
import wandb
from pathlib import Path
import re
import imageio
from collections import defaultdict
import pandas as pd
import numpy as np

import ray

import tensorflow as tf
import reverb
import sonnet as snt

import acme
from acme import core
from acme import specs
from acme import datasets
from acme import adders
from acme import wrappers
from acme.utils import counting
from acme.utils import loggers
from acme.tf import variable_utils
from acme.tf import networks as network_utils
from acme.tf import utils as tf2_utils
from acme.adders import reverb as reverb_adders

from vnl_ray.agents.learning_dmpo import DistributionalMPOLearner
from vnl_ray.agents import agent_dmpo
from vnl_ray.agents.actors import DelayedFeedForwardActor
from vnl_ray.utils import vision_rollout_and_render, rollout_and_render, render_with_rewards

# Import the proper TestPolicyWrapper from utils_tf
from vnl_ray.agents.utils_tf import TestPolicyWrapper

# Import the special evaluation wrapper
from vnl_ray.agents.utils_tf_eval import TestPolicyEvalWrapper
from vnl_ray.agents.decoder_swap_utils import swap_decoder_with_jax

# logging & plotting
from matplotlib import pyplot as plt
from io import BytesIO

from vnl_ray.agents.utils_sonnet import Sequential


# For plotting rewards
def plot_reward(idx, start_idx, rewards, terminated=False):
    """Plot the reward progress."""
    plt.figure(figsize=(6, 8))
    plt.subplot(3, 1, 1)
    x = list(range(start_idx, idx + 1))
    if terminated:
        plt.title("Episode Terminated", color="red")
    else:
        plt.title("Episode Running", color="green")

    # Sort reward keys for consistent plotting
    reward_keys = sorted(rewards.keys())
    for i, key in enumerate(reward_keys):
        plt.plot(x, rewards[key][start_idx - start_idx : idx + 1 - start_idx], label=key)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    plt.plot(x, np.cumsum(rewards.get("total", [0] * len(x))[start_idx - start_idx : idx + 1 - start_idx]))
    plt.title("Cumulative Reward")
    plt.tight_layout()
    plt.subplot(3, 1, 3)
    plt.plot(x, rewards.get("total", [0] * len(x))[start_idx - start_idx : idx + 1 - start_idx])
    plt.title("Total Reward")
    plt.tight_layout()
    canvas = plt.gcf()
    canvas.draw()
    pil_image = np.array(canvas.canvas.renderer.buffer_rgba())
    plt.close()
    return pil_image


def flatten_dict(d, parent_key="", sep="/"):
    """Recursively flattens nested dictionaries."""
    import tensorflow_probability as tfp

    items = {}
    # Handle non-dictionary inputs
    if not isinstance(d, dict):
        return {parent_key if parent_key else "value": d}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # If value is a tfp distribution, use its mean.
        if hasattr(v, "mean") and callable(getattr(v, "mean")):
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
    # Skip the first part which may fail if data is empty or malformed
    try:
        all_keys = set()
        # Use the flattened structure
        for episode_idx, activation_collection in activation_collections:
            for timestep_data in activation_collection:
                flat_data = flatten_dict(timestep_data)
                all_keys.update(flat_data.keys())
        print(f"Found {len(all_keys)} flattened keys: {all_keys}")
    except Exception as e:
        print(f"Warning during analysis: {e}")
        all_keys = set()

    # Create a minimal dataset if we can't generate full data
    if not all_keys:
        print("Creating minimal dataset due to lack of activation data")
        with pd.HDFStore(h5_file_path) as store:
            store["_empty_placeholder"] = pd.DataFrame({"info": ["No activation data collected"]})
        print(f"Created minimal placeholder dataframe in {h5_file_path}")
        return

    # Prepare container
    final_dfs = {}

    for episode_idx, activation_collection in activation_collections:
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
                # Use dimensions attribute instead of comparing array
                ndim = getattr(array, "ndim", 0)
                if ndim > 1:
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
    - Ensures that all keys from 'pose_and_trial_info', e.g. 'finger_tip_x', are captured.
    """
    directory = os.path.dirname(h5_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not kinematics_collections:
        print("No kinematics data to save")
        with pd.HDFStore(h5_file_path) as store:
            store["_empty_placeholder"] = pd.DataFrame({"info": ["No kinematics data collected"]})
        print(f"Created empty placeholder file at {h5_file_path}")
        return

    print(f"Processing kinematics data from {len(kinematics_collections)} episodes")

    try:
        geoms_positions = []
        geoms_velocities = []
        geoms_accelerations = []
        bodies_positions = []
        bodies_velocities = []
        bodies_accelerations = []
        bodies_orientations = []
        joints_angles = []
        pose_and_trial_info_list = []

        for episode_idx, kinematics_collection in kinematics_collections:
            print(f"Processing episode {episode_idx} with {len(kinematics_collection)} timesteps")
            for timestep_idx, kinematics in enumerate(kinematics_collection):
                base_dict = {"episode": episode_idx, "timestep": timestep_idx}

                # Process geoms data
                geom_pos_dict = base_dict.copy()
                geom_vel_dict = base_dict.copy()
                geom_acc_dict = base_dict.copy()
                if "geoms" in kinematics and isinstance(kinematics["geoms"], dict):
                    if "positions" in kinematics["geoms"]:
                        for geom_name, position in kinematics["geoms"]["positions"].items():
                            geom_pos_dict[f"{geom_name}"] = position
                    geoms_positions.append(geom_pos_dict)
                    if "velocities" in kinematics["geoms"]:
                        for geom_name, velocity in kinematics["geoms"]["velocities"].items():
                            geom_vel_dict[f"{geom_name}"] = velocity
                    geoms_velocities.append(geom_vel_dict)
                    if "accelerations" in kinematics["geoms"]:
                        for geom_name, acceleration in kinematics["geoms"]["accelerations"].items():
                            geom_acc_dict[f"{geom_name}"] = acceleration
                    geoms_accelerations.append(geom_acc_dict)
                else:
                    geoms_positions.append(geom_pos_dict)
                    geoms_velocities.append(geom_vel_dict)
                    geoms_accelerations.append(geom_acc_dict)

                # Process bodies data
                body_pos_dict = base_dict.copy()
                body_vel_dict = base_dict.copy()
                body_acc_dict = base_dict.copy()
                body_orient_dict = base_dict.copy()
                if "bodies" in kinematics and isinstance(kinematics["bodies"], dict):
                    if "positions" in kinematics["bodies"]:
                        for body_name, position in kinematics["bodies"]["positions"].items():
                            body_pos_dict[f"{body_name}"] = position
                    bodies_positions.append(body_pos_dict)
                    if "velocities" in kinematics["bodies"]:
                        for body_name, velocity in kinematics["bodies"]["velocities"].items():
                            body_vel_dict[f"{body_name}"] = velocity
                    bodies_velocities.append(body_vel_dict)
                    if "accelerations" in kinematics["bodies"]:
                        for body_name, acceleration in kinematics["bodies"]["accelerations"].items():
                            body_acc_dict[f"{body_name}"] = acceleration
                    bodies_accelerations.append(body_acc_dict)
                    if "orientations" in kinematics["bodies"]:
                        for body_name, orientation in kinematics["bodies"]["orientations"].items():
                            body_orient_dict[f"{body_name}"] = orientation
                    bodies_orientations.append(body_orient_dict)
                else:
                    bodies_positions.append(body_pos_dict)
                    bodies_velocities.append(body_vel_dict)
                    bodies_accelerations.append(body_acc_dict)
                    bodies_orientations.append(body_orient_dict)

                # Process joints data
                joint_dict = base_dict.copy()
                if "joints" in kinematics and isinstance(kinematics["joints"], dict):
                    for joint_name, angle in kinematics["joints"].items():
                        joint_dict[f"{joint_name}"] = angle
                joints_angles.append(joint_dict)

                # Process pose_and_trial_info data – flatten all keys if available.
                if "pose_and_trial_info" in kinematics:
                    info_dict = {}

                    def recursive_flatten(d, parent_key=""):
                        items = {}
                        for k, v in d.items():
                            new_key = f"{parent_key}{k}" if parent_key == "" else f"{parent_key}_{k}"
                            if isinstance(v, dict):
                                items.update(recursive_flatten(v, new_key))
                            else:
                                items[new_key] = v
                        return items

                    info_dict = recursive_flatten(kinematics["pose_and_trial_info"])
                    info_dict.update(base_dict)
                    pose_and_trial_info_list.append(info_dict)

        print("Converting processed data to DataFrames...")
        geoms_positions_df = pd.DataFrame(geoms_positions)
        geoms_velocities_df = pd.DataFrame(geoms_velocities)
        geoms_accelerations_df = pd.DataFrame(geoms_accelerations)
        bodies_positions_df = pd.DataFrame(bodies_positions)
        bodies_velocities_df = pd.DataFrame(bodies_velocities)
        bodies_accelerations_df = pd.DataFrame(bodies_accelerations)
        bodies_orientations_df = pd.DataFrame(bodies_orientations)
        joints_angles_df = pd.DataFrame(joints_angles)
        if pose_and_trial_info_list:
            pose_and_trial_info_df = pd.DataFrame(pose_and_trial_info_list)
        else:
            pose_and_trial_info_df = pd.DataFrame({"episode": [], "timestep": []})

        print(f"Saving DataFrames to {h5_file_path}...")
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
        print(f"Kinematics data saved successfully to {h5_file_path}")

    except Exception as e:
        print(f"Error in kinematics data processing: {e}")
        with pd.HDFStore(h5_file_path) as store:
            store["_error"] = pd.DataFrame({"error": [str(e)]})
        print(f"Saved error information to {h5_file_path}")


@dataclasses.dataclass
class DMPOConfig:
    num_actors: int = 32
    batch_size: int = 256
    prefetch_size: int = 4
    min_replay_size: int = 10_000
    max_replay_size: int = 4_000_000
    samples_per_insert: float = 32.0  # None: limiter = reverb.rate_limiters.MinSize()
    n_step: int = 5
    num_samples: int = 20
    num_learner_steps: int = 100
    clipping: bool = True
    discount: float = 0.95  # modify to align with Diego's mimic
    policy_loss_module: snt.Module | None = None
    policy_optimizer: snt.Optimizer | None = None
    critic_optimizer: snt.Optimizer | None = None
    dual_optimizer: snt.Optimizer | None = None
    target_policy_update_period: int = 101
    target_critic_update_period: int = 107
    actor_update_period: int = 1000
    logger: loggers.base.Logger | None = None
    log_every: float = 60.0  # Seconds.
    logger_save_csv_data: bool = False
    checkpoint_to_load: str | None = None  # Path to checkpoint.
    load_decoder_only: bool = False  # whether only loads decoder
    froze_decoder: bool = False  # whether we froze the weight of the decoder
    swap_decoder_with_jax: bool = False  # whether to swap decoder with JAX decoder
    decoder_h5_path: str = ""  # path to h5 file with decoder weights
    checkpoint_max_to_keep: int | None = 1  # None: keep all checkpoints.
    checkpoint_directory: str | None = "~/ray-ckpts/"  # None: no checkpointing.
    time_delta_minutes: float = 30
    terminal: str = "current_terminal"
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
    print_fn: Callable = logging.info
    userdata: dict | None = None
    actor_observation_callback: Callable | None = None
    config_dict: dict | None = None
    kickstart_teacher_cps_path: str = ("",)  # specify the location of the kickstarter teacher policy's cps
    kickstart_epsilon: float = (0.005,)
    eval_average_over: int = (200,)  # how many steps of statistic to average over in evaluator.
    KL_weights: List[float] = (0.0, 0.0)


class ReplayServer:
    """Reverb replay server, can be used with DMPO agent."""

    def __init__(self, config: DMPOConfig, environment_spec: specs.EnvironmentSpec):
        """Spawn a Reverb server with experience replay tables."""

        self._config = config
        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)
        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer,
            )

        replay_buffer = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=reverb_adders.NStepTransitionAdder.signature(environment_spec),
        )

        self._replay_server = reverb.Server(tables=[replay_buffer], port=None)
        # Get hostname and port of the server.
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        print("DEBUG: ", hostname, ip_address)
        port = self._replay_server.port
        self._replay_server_address = f"{ip_address}:{port}"

    def get_server_address(self):
        return self._replay_server_address

    def isready(self):
        """Dummy method to check if ReplayServer is ready."""
        pass


class Learner(DistributionalMPOLearner):
    """The Learning part of the DMPO agent."""

    def __init__(
        self,
        replay_server_addresses: dict,  # Allow multiple replay server address here. Modify corresponding logics.
        counter: counting.Counter,
        environment_spec: specs.EnvironmentSpec,
        dmpo_config,
        network_factory,
        label="learner",
    ):
        self._config = dmpo_config
        # self._reverb_client = reverb.Client(replay_server_address)
        self._reverb_clients = [reverb.Client(addr) for addr in replay_server_addresses.values()]
        self._label = label

        # Ensure checkpoint directory exists and is absolute
        if self._config.checkpoint_directory:
            # Force the correct path pattern
            if "/root/vast/vnl-ray/" in self._config.checkpoint_directory:
                self._config.checkpoint_directory = self._config.checkpoint_directory.replace(
                    "/root/vast/vnl-ray/", "/root/vast/eric/vnl-ray/"
                )
                print(f"Corrected checkpoint path to: {self._config.checkpoint_directory}")
            elif not self._config.checkpoint_directory.startswith("/root/vast/eric/vnl-ray/"):
                # Hardcode the path if it doesn't already have the correct prefix
                self._config.checkpoint_directory = "/root/vast/eric/vnl-ray/training/ray-mouse-mouse_reach-ckpts/"
                print(f"Reset to hardcoded checkpoint path: {self._config.checkpoint_directory}")

            os.makedirs(self._config.checkpoint_directory, exist_ok=True)
            print(f"Ensuring checkpoint directory exists: {self._config.checkpoint_directory}")

        def wrapped_network_factory(action_spec):
            networks_dict = network_factory(action_spec)
            networks = agent_dmpo.DMPONetworks(
                policy_network=networks_dict.get("policy"),
                critic_network=networks_dict.get("critic"),
                observation_network=networks_dict.get(
                    "observation", tf.identity
                ),  # optionally use the user defined observation network
                # if none is define, use the identity function.
            )
            return networks

        # Create the networks to optimize (online) and target networks.
        online_networks = wrapped_network_factory(environment_spec.actions)
        target_networks = copy.deepcopy(online_networks)
        # Initialize the networks.
        online_networks.init(environment_spec)
        target_networks.init(environment_spec)

        # Check if we should swap the decoder with a JAX decoder
        if dmpo_config.swap_decoder_with_jax and dmpo_config.decoder_h5_path:
            # This assumes the policy network has an 'intention_network' attribute
            # that contains the intention network with decoder
            if hasattr(online_networks.policy_network, "intention_network"):
                #                 print(f"\n======================================================")
                #                 print(f"SWAPPING DECODER: Replacing standard decoder with JAX decoder")
                #                 print(f"H5 CHECKPOINT PATH: {dmpo_config.decoder_h5_path}")
                #                 print(f"======================================================\n")

                decoder_layer_sizes = dmpo_config.userdata.get("decoder_layer_sizes", [512, 512, 512])
                if (
                    "learner_network" in dmpo_config.userdata.get("config", {})
                    and "decoder_layer_sizes" in dmpo_config.userdata["config"]["learner_network"]
                ):
                    decoder_layer_sizes = dmpo_config.userdata["config"]["learner_network"]["decoder_layer_sizes"]

                #                 print(f"Using decoder layer sizes: {decoder_layer_sizes}")

                swap_decoder_with_jax(
                    online_networks.policy_network.intention_network,
                    decoder_h5_path=dmpo_config.decoder_h5_path,
                    action_size=environment_spec.actions.shape[0],
                    layer_sizes=decoder_layer_sizes,
                    min_scale=0.1,
                    freeze_weights=dmpo_config.froze_decoder,
                )

                # Also swap the target network's decoder
                if hasattr(target_networks.policy_network, "intention_network"):
                    #                     print(f"Also swapping target network decoder...")
                    swap_decoder_with_jax(
                        target_networks.policy_network.intention_network,
                        decoder_h5_path=dmpo_config.decoder_h5_path,
                        action_size=environment_spec.actions.shape[0],
                        layer_sizes=decoder_layer_sizes,
                        min_scale=0.1,
                        freeze_weights=dmpo_config.froze_decoder,
                    )

        # print(f"\n✓ DECODER SWAP SUCCESSFULLY COMPLETED FOR ALL NETWORKS")
        # print(f"======================================================\n")

        datasets = [
            self._make_dataset_iterator(c) for c in self._reverb_clients
        ]  # (SY) add multiple reverbe client here
        counter = counting.Counter(parent=counter, prefix=label)
        if self._config.logger is None:
            logger = loggers.make_default_logger(
                label=label,
                time_delta=self._config.log_every,
                steps_key=f"{label}_steps",
                print_fn=self._config.print_fn,
                save_data=self._config.logger_save_csv_data,
            )
        else:
            if "logger_kwargs" in self._config.userdata:
                logger_kwargs = self._config.userdata["logger_kwargs"]
            else:
                logger_kwargs = {}
            logger = self._config.logger(
                label=label,
                time_delta=self._config.log_every,
                wandb_project=True,
                identity="learner",
                **logger_kwargs,
            )

        # Maybe checkpoint and snapshot the learner (saved in ~/acme/).
        checkpoint_enable = self._config.checkpoint_directory is not None
        if checkpoint_enable:
            print(f"Checkpointing enabled. Saving to: {self._config.checkpoint_directory}")
            print(f"Checkpoint interval: {self._config.time_delta_minutes} minutes")
        else:
            print("WARNING: Checkpointing is disabled!")

        # Have to call superclass constructor in this way.
        # Solved with Ray issue:  https://github.com/ray-project/ray/issues/449
        DistributionalMPOLearner.__init__(
            self,
            policy_network=online_networks.policy_network,
            critic_network=online_networks.critic_network,
            observation_network=online_networks.observation_network,
            target_policy_network=target_networks.policy_network,
            target_critic_network=target_networks.critic_network,
            target_observation_network=target_networks.observation_network,
            policy_loss_module=self._config.policy_loss_module,
            policy_optimizer=self._config.policy_optimizer,
            critic_optimizer=self._config.critic_optimizer,
            dual_optimizer=self._config.dual_optimizer,
            clipping=self._config.clipping,
            discount=self._config.discount,
            num_samples=self._config.num_samples,
            target_policy_update_period=self._config.target_policy_update_period,
            target_critic_update_period=self._config.target_critic_update_period,
            datasets=datasets,
            logger=logger,
            counter=counter,
            checkpoint_enable=checkpoint_enable,
            checkpoint_max_to_keep=self._config.checkpoint_max_to_keep,
            directory=self._config.checkpoint_directory,
            checkpoint_to_load=self._config.checkpoint_to_load,
            time_delta_minutes=self._config.time_delta_minutes,
            kickstart_teacher_cps_path=self._config.kickstart_teacher_cps_path,
            kickstart_epsilon=self._config.kickstart_epsilon,
            replay_server_addresses=replay_server_addresses,
            KL_weights=self._config.KL_weights,
            load_decoder_only=self._config.load_decoder_only,
            froze_decoder=self._config.froze_decoder,
        )

        # Verify checkpoint directories after initialization
        if self._checkpointer is not None:
            print(f"Checkpointer initialized with directory: {self._checkpointer._checkpoint_dir}")
            if not os.path.exists(self._checkpointer._checkpoint_dir):
                print(f"WARNING: Checkpointer directory does not exist: {self._checkpointer._checkpoint_dir}")
            elif not os.access(self._checkpointer._checkpoint_dir, os.W_OK):
                print(f"WARNING: Checkpointer directory not writable: {self._checkpointer._checkpoint_dir}")
            else:
                print(f"Checkpointer directory exists and is writable: {self._checkpointer._checkpoint_dir}")

    def _step(self, iterator):
        # Workaround to access _step in DistributionalMPOLearner:
        # @tf.function
        # def _step(self)
        #    ...
        return DistributionalMPOLearner._step(self, iterator)

    def run(self, num_steps=None):
        del num_steps  # Not used.
        # Run fixed number of learning steps and return control to have a chance
        # to process calls to `get_variables`.
        for _ in range(self._config.num_learner_steps):
            self.step()

    def isready(self):
        """Dummy method to check if learner is ready."""
        pass

    def get_checkpoint_dir(self):
        """Return Checkpointer and Snapshotter directories, if any."""
        if self._checkpointer is not None:
            checkpointer_dir = self._checkpointer._checkpoint_dir
            snapshotter_dir = self._snapshotter.directory if self._snapshotter else None
            print(f"Current checkpointer directory: {checkpointer_dir}")
            print(f"Current snapshotter directory: {snapshotter_dir}")
            return checkpointer_dir, snapshotter_dir
        print("WARNING: No checkpointer available!")
        return None, None

    def _make_dataset_iterator(
        self,
        reverb_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=reverb_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size,
        )
        return iter(dataset)

    def get_policy(self):
        """Return the policy network for evaluation purposes."""
        print("Returning policy network for evaluation")
        # Return the policy network that was restored from the checkpoint
        try:
            # Create a lightweight copy that can be transported over Ray
            # This helps avoid serialization issues with TF objects
            if hasattr(self, "_policy_network"):
                print(f"Found policy network of type: {type(self._policy_network).__name__}")
                return self._policy_network
            else:
                print("WARNING: _policy_network not found!")
                return None
        except Exception as e:
            print(f"Error in get_policy: {e}")
            raise

    def restore(self, checkpoint_path):
        try:
            # Load checkpoint and handle missing keys
            checkpoint = tf.train.Checkpoint(policy_network=self._policy_network)
            checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
            print(f"Checkpoint restored successfully from {checkpoint_path}")
        except KeyError as e:
            print(f"Error restoring checkpoint: {e}")
            print("Ensure the checkpoint matches the expected structure.")


def create_eval_policy_network(networks, deterministic=True, debug=False):
    """Creates a properly wrapped policy network for evaluation with IntentionNetwork support.

    This function creates a properly wrapped network that handles the flow:
    observation → observation_network → policy_network (encoder→intention→decoder) → stochastic head

    Args:
        networks: DMPONetworks object containing policy and observation networks
        deterministic: Whether to use deterministic (mean) or stochastic sampling
        debug: Whether to print debug information about the network structure

    Returns:
        A Sequential network that can be directly called with observations
    """
    if debug:
        print(f"Creating eval policy network with deterministic={deterministic}")
        print(f"Policy network type: {type(networks.policy_network).__name__}")

        # Check if we're dealing with an IntentionNetwork
        if hasattr(networks.policy_network, "encoder") and hasattr(networks.policy_network, "decoder"):
            print("IntentionNetwork architecture detected")

            # Print additional debugging info about the intention network
            print(f"Intention size: {getattr(networks.policy_network, 'intention_size', 'unknown')}")
            print(f"Task obs size: {getattr(networks.policy_network, 'task_obs_size', 'unknown')}")

            if hasattr(networks.policy_network, "encoder"):
                print("Encoder network present")
            if hasattr(networks.policy_network, "decoder"):
                print("Decoder network present")

    # Create the appropriate stochastic head based on deterministic flag
    if deterministic:
        # For evaluation, use the mean of the action distribution
        stochastic_layer = network_utils.StochasticMeanHead()
    else:
        # For actors, sample from the action distribution
        stochastic_layer = network_utils.StochasticSamplingHead()

    # Create a sequential network that properly processes observations through the network
    return Sequential(
        [
            networks.observation_network,
            networks.policy_network,
            stochastic_layer,
        ]
    )


class EnvironmentLoop(acme.EnvironmentLoop):
    """Actor and Evaluator class."""

    # Add a class-level variable to persist the fixed snapshot for evaluators.
    _global_fixed_snapshot = None

    def __init__(
        self,
        replay_server_address: str,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        network_factory,
        environment_factory,
        dmpo_config,
        actor_or_evaluator="actor",
        label=None,
        ray_head_node_ip: str | None = None,
        egl_device_id_head_node: list | None = None,
        egl_device_id_worker_node: list | None = None,
        task_name: str = "",
        snapshotter_dir: str | None = None,
        checkpoint_snapshot_dir: str | None = None,
        force_render: bool = False,
        snapshot_path: str | None = None,
        restored_policy: any = None,
    ):
        """The actor process."""
        # Debug: Print all received parameters immediately on entry
        print(f"\n=== PARAMETER DEBUGGING in EnvironmentLoop.__init__ ===")
        print(f"task_name: {task_name}")
        print(f"actor_or_evaluator: {actor_or_evaluator}")
        print(f"force_render: {force_render}")
        print(f"snapshot_path (raw): {snapshot_path}")
        print(f"snapshot_path (type): {type(snapshot_path)}")
        print(f"checkpoint_snapshot_dir: {checkpoint_snapshot_dir}")

        # Store attributes - IMPORTANT: Use self.actor_or_evaluator (no underscore) to avoid potential conflicts
        self.actor_or_evaluator = actor_or_evaluator
        self._force_render = force_render
        self._task_name = task_name
        self._episode_num = 0

        # For evaluator processes, persist the snapshot path as a fixed snapshot.
        if actor_or_evaluator == "evaluator":
            if isinstance(snapshot_path, str) and snapshot_path and os.path.exists(snapshot_path):
                self._snapshot_path = snapshot_path
                EnvironmentLoop._global_fixed_snapshot = snapshot_path
                self._fixed_snapshot = Path(snapshot_path)
                print(f"Fixed evaluator snapshot set to: {self._fixed_snapshot}")
            elif EnvironmentLoop._global_fixed_snapshot is not None:
                self._snapshot_path = EnvironmentLoop._global_fixed_snapshot
                self._fixed_snapshot = Path(EnvironmentLoop._global_fixed_snapshot)
                print(f"Using global fixed evaluator snapshot: {self._fixed_snapshot}")
            else:
                self._snapshot_path = None
                self._fixed_snapshot = None
                print("WARNING: Evaluator fixed snapshot not set!")
        else:
            # Actors do not use a snapshot.
            self._snapshot_path = None
            self._fixed_snapshot = None

        # Persist the initial snapshot path for later restoration
        self._init_snapshot_path = self._snapshot_path

        # Only initialize collections if we're explicitly collecting data with force_render
        if force_render:
            self._all_activation_collections = []
            self._all_kinematics_collections = []
        self._actuator_type = getattr(
            dmpo_config.userdata.get("config", {}).get("run_config", {}), "actuator_type", "torque"
        )

        # Maybe adjust EGL_DEVICE_ID environment variable internally in actor.
        if ray_head_node_ip is not None:
            current_node_id = ray.get_runtime_context().node_id.hex()
            running_on_head_node = False
            for node in ray.nodes():
                if node["NodeID"] == current_node_id and node["NodeManagerAddress"] == ray_head_node_ip:
                    running_on_head_node = True
                    break
            if running_on_head_node:
                egl_device_id = np.random.choice(egl_device_id_head_node)
            else:
                egl_device_id = np.random.choice(egl_device_id_worker_node)
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_id)

        assert actor_or_evaluator in ["actor", "evaluator"]
        if actor_or_evaluator == "actor":
            self._reverb_client = reverb.Client(replay_server_address)
        else:
            self._reverb_client = None

        self._config = dmpo_config
        # Create the environment.
        environment = environment_factory()
        environment_spec = specs.make_environment_spec(environment)

        def wrapped_network_factory(action_spec):
            networks_dict = network_factory(action_spec)
            networks = agent_dmpo.DMPONetworks(
                policy_network=networks_dict.get("policy"),
                critic_network=networks_dict.get("critic"),
                observation_network=networks_dict.get("observation", tf.identity),
            )
            return networks

        # Create the policy network, adder, ...
        networks = wrapped_network_factory(environment_spec.actions)
        networks.init(environment_spec)

        if actor_or_evaluator == "actor":
            # Actor: sample from policy_network distribution.
            policy_network = create_eval_policy_network(networks, deterministic=False, debug=True)
            adder = self._make_adder(self._reverb_client)
            save_data = False

        elif actor_or_evaluator == "evaluator":
            # Evaluator: get mean from policy_network distribution.
            policy_network = create_eval_policy_network(networks, deterministic=True, debug=True)
            adder = None
            save_data = self._config.logger_save_csv_data

        # Create the agent.
        actor = self._make_actor(
            policy_network=policy_network,
            adder=adder,
            variable_source=variable_source,
            observation_callback=self._config.actor_observation_callback,
        )

        # Create logger and counter; actors will not spam bigtable.
        if label is None:
            label = actor_or_evaluator  # Set default label if none provided.
        if self._config.logger is None:
            logger = loggers.make_default_logger(
                label=label,
                save_data=save_data,
                time_delta=self._config.log_every,
                steps_key=actor_or_evaluator + "_steps",
                print_fn=self._config.print_fn,
            )
        else:
            if "logger_kwargs" in self._config.userdata:
                logger_kwargs = self._config.userdata["logger_kwargs"]
            else:
                logger_kwargs = {}
            if actor_or_evaluator == "evaluator":
                print(f"Evaluator Node for Logger! Task Name: {task_name}")
            logger = self._config.logger(
                label=label,
                time_delta=self._config.log_every,
                # only create project for evaluators,
                wandb_project=actor_or_evaluator == "evaluator",
                identity="evaluator",
                task_name=task_name,
                **logger_kwargs,
            )

        if snapshotter_dir is not None:
            self._snapshotter_dir = Path(snapshotter_dir)
        else:
            self._snapshotter_dir = None

        # Store the checkpoint snapshot directory if provided
        if checkpoint_snapshot_dir is not None:
            self._checkpoint_snapshot_dir = Path(checkpoint_snapshot_dir)
        else:
            self._checkpoint_snapshot_dir = None

        self._latest_snapshot = None
        self._highest_snap_num = -1
        self._task_name = task_name
        self._environment_factory = environment_factory

        # for evaluator logging support for mean
        self._stats = []

        self._restored_policy = restored_policy  # store the provided restored policy

        super().__init__(environment, actor, counter, logger)

    def run_episode(self) -> loggers.LoggingData:
        """Add rendering support for evaluator, and added aggregate stats"""
        # Only increment episode counter if we're collecting episode data
        if self._force_render:
            self._episode_num += 1

        try:
            logging_data = super().run_episode()
        except Exception as e:
            print(f"Exception: {e} encountered in run_episode. Returned Null result for this episode.")
            return {"episode_length": 0, "episode_return": 0, "steps_per_second": 0}

        # IMPORTANT: Use self.actor_or_evaluator (no underscore)
        if self.actor_or_evaluator == "evaluator":
            self._stats.append(logging_data)
            # self.stats is a [t1_data, t2_data, ...] array.
            if len(self._stats) >= self._config.eval_average_over:  # only report summary statistic one a while
                self._stats.pop(0)  # pop out the stats

            # Use enhanced rendering if explicitly requested with force_render
            if hasattr(self, "_force_render") and self._force_render:
                self.load_snapshot_and_render_with_data(logging_data)
            else:
                # Use original rendering approach for normal training
                self.load_snapshot_and_render(logging_data)

            logging_data.update(self._eval_agg_stat(True))  # update in place

        return logging_data

    def load_snapshot_and_render(self, logging_data):
        """
        Check the snapshot directory, renders whenever there is a
        new policy snapshot, optionally send it to wandb. Modify the logging_data dict in place
        """
        # Only attempt rendering if we have a snapshotter directory
        if not hasattr(self, "_snapshotter_dir") or self._snapshotter_dir is None:
            return

        render = False
        for path in self._snapshotter_dir.iterdir():
            match = re.match(r"policy-(\d+)", path.name)  # Look for the pattern "policy-number"
            if match:
                number = int(match.group(1))
                if number > self._highest_snap_num:
                    self._highest_snap_num = number
                    self._latest_snapshot = path
                    render = True

        if render:
            videos_path = self._snapshotter_dir.parent / "videos"
            videos_path.mkdir(parents=True, exist_ok=True)
            rendering_path = os.path.join(str(videos_path), f"{self._task_name}-{self._highest_snap_num}.mp4")

            try:
                policy = tf.saved_model.load(str(self._latest_snapshot))
                policy = TestPolicyWrapper(policy)
            except OSError as e:
                print(f"Policy Loading Error: {e}. Skipping rendering for this policy.")
                self._highest_snap_num -= 1  # retry rendering the next iter.
                return

            # Create environment for rendering
            env = self._environment_factory()
            env = wrappers.SinglePrecisionWrapper(env)
            env = wrappers.CanonicalSpecWrapper(env, clip=False)

            # Use original render_with_rewards from vnl_ray.utils
            frames = render_with_rewards(env, policy, rollout_length=300)

            with imageio.get_writer(rendering_path, fps=1 / env.control_timestep()) as video:
                for f in frames:
                    video.append_data(f)

            logging_data["rollout"] = wandb.Video(rendering_path, format="mp4")

    def load_snapshot_and_render_with_data(self, logging_data):
        """
        Enhanced rendering that collects activations and kinematics data
        and saves it to an h5 file.
        """
        print("\n=== SNAPSHOT DIRECTORY DEBUGGING ===")
        print(f"force_render flag is set to: {self._force_render}")
        print(f"Task name: {self._task_name}")
        print(f"Episode number: {self._episode_num}")

        # Always use the fixed snapshot for evaluators.
        if self._fixed_snapshot is not None:
            self._latest_snapshot = self._fixed_snapshot
            print(f"Using fixed snapshot: {self._latest_snapshot}")
            match = re.search(r"policy-(\d+)", str(self._fixed_snapshot))
            if match:
                self._highest_snap_num = int(match.group(1))
                print(f"Using snapshot number: {self._highest_snap_num}")
            else:
                self._highest_snap_num = 0
        else:
            print("ERROR: No fixed snapshot available, cannot render.")
            return

        # Use the restored policy if provided; otherwise, load as before.
        if self._restored_policy is not None:
            policy = self._restored_policy
            print("Using restored policy passed from training instance.")
        else:
            raise ValueError("No restored policy was provided; evaluator cannot run without it.")

        # Use the special evaluation wrapper that avoids copy() calls
        policy = TestPolicyEvalWrapper(policy, debug=True)
        print("Using TestPolicyEvalWrapper for evaluation")

        # Set up the environment for rendering.
        env = self._environment_factory()
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env, clip=False)

        # Collect frames, activations, and kinematics.
        frames, activations, kinematics = self._collect_rendering_data(
            env, policy, rollout_length=300, episode_num=self._episode_num
        )

        # (Optional) accumulate data collections.
        if hasattr(self, "_all_activation_collections"):
            self._all_activation_collections.append((self._episode_num, activations))
        if hasattr(self, "_all_kinematics_collections"):
            self._all_kinematics_collections.append((self._episode_num, kinematics))

        # Determine where to save the rendered video.
        if (
            self._force_render
            and hasattr(self, "_checkpoint_snapshot_dir")
            and self._checkpoint_snapshot_dir is not None
        ):
            videos_dir = self._checkpoint_snapshot_dir.parent / "videos"
        elif hasattr(self, "_snapshotter_dir") and self._snapshotter_dir is not None:
            videos_dir = self._snapshotter_dir.parent / "videos"
        else:
            print("No directory available to save videos")
            return

        videos_dir.mkdir(parents=True, exist_ok=True)
        rendering_path = os.path.join(
            str(videos_dir), f"{self._task_name}-ep{self._episode_num}-{self._highest_snap_num}.mp4"
        )

        # Write the video.
        with imageio.get_writer(rendering_path, fps=1 / env.control_timestep()) as video:
            for f in frames:
                video.append_data(f)

        logging_data["rollout"] = wandb.Video(rendering_path, format="mp4")
        print(f"Saved video to {rendering_path}")

        # (Optional) Save activation/kinematics data.
        if hasattr(self, "_all_activation_collections") and hasattr(self, "_all_kinematics_collections"):
            data_dir = videos_dir.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            # Try to save activation data with error handling
            try:
                activations_file = os.path.join(
                    str(data_dir), f"{self._task_name}-activations-{self._highest_snap_num}.h5"
                )
                process_and_save_activation_collection(self._all_activation_collections, activations_file)
                print(f"Saved activation data to {activations_file}")
            except Exception as e:
                print(f"Error saving activation data: {e}")

            # Try to save kinematics data with error handling
            try:
                kinematics_file = os.path.join(
                    str(data_dir), f"{self._task_name}-kinematics-{self._highest_snap_num}.h5"
                )
                process_and_save_kinematics_collection(self._all_kinematics_collections, kinematics_file)
                print(f"Saved kinematics data to {kinematics_file}")
            except Exception as e:
                print(f"Error saving kinematics data: {e}")

            print(f"Data collection process complete")

    def _extract_pose_info(self, env, episode_idx, timestep_idx):
        """Extract pose and trial info from environment safely without ambiguous truth value checks."""
        # Basic metadata
        info = {
            "episode_number": episode_idx,
            "index": timestep_idx,
        }

        # Create a safer helper function to avoid ambiguous truth value errors
        def safe_check_in(key, container):
            """Safely check if key is in container without triggering ambiguous truth value errors."""
            try:
                # Convert both to strings to be absolutely safe
                str_key = str(key)
                container_keys = [str(k) for k in container]
                return str_key in container_keys
            except Exception:
                return False

        def safe_get_array_item(arr, index, default=None):
            """Safely get item from array at index without triggering ambiguous truth value errors."""
            try:
                if arr is None:
                    return default
                arr_np = np.asarray(arr)
                if index < arr_np.size:
                    return float(arr_np.flat[index])
                return default
            except Exception:
                return default

        # For mouse_reach environment, extract detailed kinematics data
        try:
            physics = env.physics

            # Get finger tip position - with safer checks
            try:
                # Check if geom_xpos exists
                if hasattr(physics.named.data, "geom_xpos"):
                    # Get all keys in a list to avoid ambiguous truth value checks
                    all_keys = list(
                        physics.named.data.geom_xpos.keys() if hasattr(physics.named.data.geom_xpos, "keys") else []
                    )

                    # Check keys safely with string comparison
                    if "mouse/finger_tip" in all_keys:
                        finger_pos = physics.named.data.geom_xpos["mouse/finger_tip"]
                        info.update(
                            {
                                "finger_tip_x": safe_get_array_item(finger_pos, 0, 0.0),
                                "finger_tip_y": safe_get_array_item(finger_pos, 1, 0.0),
                                "finger_tip_z": safe_get_array_item(finger_pos, 2, 0.0),
                            }
                        )
                        print(f"Found finger_tip position via direct named access")
                    elif "finger_tip" in all_keys:
                        finger_pos = physics.named.data.geom_xpos["finger_tip"]
                        info.update(
                            {
                                "finger_tip_x": safe_get_array_item(finger_pos, 0, 0.0),
                                "finger_tip_y": safe_get_array_item(finger_pos, 1, 0.0),
                                "finger_tip_z": safe_get_array_item(finger_pos, 2, 0.0),
                            }
                        )
                        print(f"Found finger_tip position via direct named access (no prefix)")
            except Exception as e:
                print(f"Could not get finger_tip position via named access: {e}")
                try:
                    # Try site position with safer checks
                    if hasattr(physics.named.data, "site_xpos"):
                        all_site_keys = list(
                            physics.named.data.site_xpos.keys() if hasattr(physics.named.data.site_xpos, "keys") else []
                        )

                        if "mouse/finger_tip" in all_site_keys:
                            finger_pos = physics.named.data.site_xpos["mouse/finger_tip"]
                            info.update(
                                {
                                    "finger_tip_x": safe_get_array_item(finger_pos, 0, 0.0),
                                    "finger_tip_y": safe_get_array_item(finger_pos, 1, 0.0),
                                    "finger_tip_z": safe_get_array_item(finger_pos, 2, 0.0),
                                }
                            )
                            print(f"Found finger_tip position via site_xpos")
                        elif "finger_tip" in all_site_keys:
                            finger_pos = physics.named.data.site_xpos["finger_tip"]
                            info.update(
                                {
                                    "finger_tip_x": safe_get_array_item(finger_pos, 0, 0.0),
                                    "finger_tip_y": safe_get_array_item(finger_pos, 1, 0.0),
                                    "finger_tip_z": safe_get_array_item(finger_pos, 2, 0.0),
                                }
                            )
                            print(f"Found finger_tip position via site_xpos (no prefix)")
                except Exception as site_e:
                    print(f"Could not get finger_tip position via site_xpos: {site_e}")

            # Get target position
            try:
                if hasattr(physics.named.data, "geom_xpos"):
                    all_geom_keys = list(
                        physics.named.data.geom_xpos.keys() if hasattr(physics.named.data.geom_xpos, "keys") else []
                    )

                    if "mouse/target" in all_geom_keys:
                        target_pos = physics.named.data.geom_xpos["mouse/target"]
                        info.update(
                            {
                                "target_position_x": safe_get_array_item(target_pos, 0, 0.0),
                                "target_position_y": safe_get_array_item(target_pos, 1, 0.0),
                                "target_position_z": safe_get_array_item(target_pos, 2, 0.0),
                            }
                        )
                    elif "target" in all_geom_keys:
                        target_pos = physics.named.data.geom_xpos["target"]
                        info.update(
                            {
                                "target_position_x": safe_get_array_item(target_pos, 0, 0.0),
                                "target_position_y": safe_get_array_item(target_pos, 1, 0.0),
                                "target_position_z": safe_get_array_item(target_pos, 2, 0.0),
                            }
                        )
            except Exception as e:
                print(f"Could not get target position via geom_xpos: {e}")
                try:
                    if hasattr(physics.named.data, "site_xpos"):
                        all_site_keys = list(
                            physics.named.data.site_xpos.keys() if hasattr(physics.named.data.site_xpos, "keys") else []
                        )

                        if "mouse/target" in all_site_keys:
                            target_pos = physics.named.data.site_xpos["mouse/target"]
                            info.update(
                                {
                                    "target_position_x": safe_get_array_item(target_pos, 0, 0.0),
                                    "target_position_y": safe_get_array_item(target_pos, 1, 0.0),
                                    "target_position_z": safe_get_array_item(target_pos, 2, 0.0),
                                }
                            )
                        elif "target" in all_site_keys:
                            target_pos = physics.named.data.site_xpos["target"]
                            info.update(
                                {
                                    "target_position_x": safe_get_array_item(target_pos, 0, 0.0),
                                    "target_position_y": safe_get_array_item(target_pos, 1, 0.0),
                                    "target_position_z": safe_get_array_item(target_pos, 2, 0.0),
                                }
                            )
                except Exception as site_e:
                    print(f"Could not get target position via site_xpos: {site_e}")

            # Get joint angles with proper error handling for different MuJoCo versions
            try:
                # Try accessing through named.data.qpos first (works in some MuJoCo versions)
                if hasattr(physics.named.data, "qpos"):
                    qpos_dict = physics.named.data.qpos
                    # Check if qpos is a dict-like object with items() method
                    if hasattr(qpos_dict, "items") and callable(getattr(qpos_dict, "items")):
                        for joint_name, value in qpos_dict.items():
                            info[f"joint_{joint_name}"] = float(value)
                        print("Successfully accessed joint angles through named.data.qpos")
                    else:
                        # qpos exists but might be a numpy array or other non-dict object
                        print("physics.named.data.qpos exists but is not a dictionary")

                # If the above approach didn't work, fall back to manual approach
                joint_keys = [k for k in info.keys() if k.startswith("joint_")]
                if not joint_keys:  # Only proceed if we don't have joint data yet
                    print("Falling back to direct indexing of qpos")
                    # Direct access with indices
                    if hasattr(physics.data, "qpos") and hasattr(physics.data.qpos, "__len__"):
                        for i in range(len(physics.data.qpos)):
                            info[f"joint_{i}"] = float(physics.data.qpos[i])
            except Exception as e:
                print(f"Exception while extracting joint data: {e}")
                # Fallback to just storing raw qpos data
                try:
                    qpos = physics.data.qpos
                    if qpos is not None and hasattr(qpos, "__len__") and len(qpos) > 0:
                        for i in range(len(qpos)):
                            info[f"joint_{i}"] = float(qpos[i])
                except Exception as qpos_e:
                    print(f"Failed even basic qpos extraction: {qpos_e}")

        except Exception as e:
            print(f"Error extracting physics data: {e}")
            # Add environment-specific debugging info
            info["_error"] = str(e)

        return info

    def _collect_rendering_data(self, env, policy, rollout_length=300, episode_num=0):
        """Collect frames, activations, and kinematics during rollout."""
        # Initialize collections
        frames = []
        activations = []
        kinematics = []

        # Function to safely check if array/tensor has elements
        def safe_has_elements(arr):
            """Safely check if an array has elements without triggering truth value ambiguity."""
            import numpy as np

            if arr is None:
                return False
            try:
                # Convert to numpy array if it isn't already
                np_arr = np.asarray(arr)
                # Check size attribute which works for all numpy arrays
                return np_arr.size > 0
            except (TypeError, ValueError, AttributeError):
                # Fall back to safe conversion to bool for non-array objects
                try:
                    return bool(arr)
                except (ValueError, TypeError):
                    return False

        # Reset environment and get initial observation
        timestep = env.reset()

        # Debug observation format more clearly
        if isinstance(timestep.observation, dict):
            print(f"DEBUG: Initial observation is a dictionary with keys: {list(timestep.observation.keys())}")
            for k, v in timestep.observation.items():
                print(f"  {k}: shape={np.asarray(v).shape}, dtype={type(v)}")
        else:
            print(
                f"DEBUG: Initial observation is a {type(timestep.observation)} with shape {np.asarray(timestep.observation).shape}"
            )

        # Get named entities from physics model
        try:
            geom_names = [env.physics.model.id2name(i, "geom") for i in range(env.physics.model.ngeom)]
            body_names = [env.physics.model.id2name(i, "body") for i in range(env.physics.model.nbody)]
            joint_names = [env.physics.model.id2name(i, "joint") for i in range(env.physics.model.njnt)]
        except Exception as e:
            print(f"Error getting names from physics model: {e}")
            geom_names = []
            body_names = []
            joint_names = []

        # For velocity calculations via finite differences
        prev_geom_positions = {}
        prev_body_positions = {}
        prev_finger_position = None

        # Save initial physics state with safe tensor handling
        try:
            # Collect initial kinematics data
            kinematics_data = self._extract_pose_info(env, episode_num, 0)  # Extract all pose info via helper
            kinematics.append(kinematics_data)
        except Exception as e:
            print(f"Error collecting initial kinematics: {e}")
            kinematics.append({"error": str(e), "episode_number": episode_num, "index": 0})

        for i in range(rollout_length):
            # Get observation and render frame
            observation = timestep.observation
            frame = env.physics.render(camera_id=0)
            frames.append(frame)

            # Process observation for policy
            try:
                # Always convert to numpy arrays to avoid any tensor issues
                if isinstance(observation, dict):
                    numpy_observation = {k: np.array(v) for k, v in observation.items()}
                    policy_input = numpy_observation
                else:
                    numpy_observation = np.array(observation)
                    policy_input = numpy_observation

                # Get action and activations from policy
                try:
                    policy_output = policy(policy_input, return_activations=True)
                    action, policy_activations = policy_output
                    activations.append({"episode": episode_num, "timestep": i, "data": policy_activations})
                except Exception as e:
                    print(f"Error during policy evaluation: {e}")
                    # Create a default action if policy failed
                    action = np.zeros(env.action_spec().shape)
                    activations.append({"episode": episode_num, "timestep": i, "data": {"error": str(e)}})

                # Ensure action is properly shaped for environment
                if len(action.shape) > 1 and action.shape[0] == 1:
                    action = action[0]

            except Exception as e:
                print(f"Error processing observation for policy: {e}")
                action = np.zeros(env.action_spec().shape)

            # Take action in environment
            timestep = env.step(action)

            # Collect kinematics data after the step using our helper method
            try:
                kinematics_data = self._extract_pose_info(env, episode_num, i + 1)
                kinematics.append(kinematics_data)
            except Exception as e:
                print(f"Error collecting kinematics at step {i}: {e}")
                kinematics.append({"error": str(e), "episode_number": episode_num, "index": i + 1})

            # Check if episode ended
            if timestep.last():
                print(f"Episode ended at step {i}")
                break

        print(f"Collected data for {len(frames)} frames, {len(activations)} activations, {len(kinematics)} kinematics")
        return frames, activations, kinematics

    def isready(self):
        """Dummy method to check if actor is ready."""
        pass

    def _make_actor(
        self,
        policy_network: snt.Module,
        adder: adders.Adder | None = None,
        variable_source: core.VariableSource | None = None,
        observation_callback: Callable | None = None,
    ):
        """Create an actor instance."""
        if variable_source:
            # Create the variable client responsible for keeping the actor up-to-date.
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={"policy": policy_network.variables},
                update_period=self._config.actor_update_period,  # was: hard-coded 1000,
            )
            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()
        else:
            variable_client = None

        # This is a modified version of actors.FeedForwardActor in Acme.
        return DelayedFeedForwardActor(
            policy_network=policy_network,
            adder=adder,
            variable_client=variable_client,
            action_delay=None,
            observation_callback=observation_callback,
        )

    def _make_adder(self, replay_client: reverb.Client) -> adders.Adder:
        """Create an adder which records data generated by the actor/environment."""
        return reverb_adders.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: lambda x: 1.0},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount,
        )

    def _eval_agg_stat(self, include_raw=False) -> loggers.LoggingData:
        """
        For evaluators, calculates the aggregate statistics such as
        avg episode return, avg episode length, and avg sps
        """
        agg = {}
        stats_key = ["episode_length", "episode_return"]
        if len(self._stats) >= self._config.eval_average_over:  # only report summary statistic one a while
            avg = {
                f"avg_{key}": np.mean([d[key] for d in self._stats])
                for key in ["episode_length", "episode_return", "steps_per_second"]
                if all(key in d for d in self._stats)
            }
            var = {
                f"var_{key}": np.var([d[key] for d in self._stats])
                for key in stats_key
                if all(key in d for d in self._stats)
            }
            maxi = {
                f"max_{key}": np.max([d[key] for d in self._stats])
                for key in stats_key
                if all(key in d for d in self._stats)
            }
            mini = {
                f"min_{key}": np.min([d[key] for d in self._stats])
                for key in stats_key
                if all(key in d for d in self._stats)
            }
            agg.update(avg)
            agg.update(var)
            agg.update(maxi)
            agg.update(mini)
            if include_raw:
                for key in stats_key:
                    if all(key in d for d in self._stats):
                        agg[f"curr_{key}"] = np.array([d[key] for d in self._stats])
        return agg

    def verify_snapshot_path(self, expected_path):
        """Verify that the snapshot path is set correctly."""
        if self._fixed_snapshot is not None:
            actual_path = str(self._fixed_snapshot)
            print(f"Verifying snapshot path: expected={expected_path}, actual={actual_path}")
            is_match = actual_path == expected_path
            print(f"Snapshot path verification: {'SUCCESS' if is_match else 'FAILED'}")
            return is_match
        else:
            print("ERROR: No fixed snapshot is set")
            return False
