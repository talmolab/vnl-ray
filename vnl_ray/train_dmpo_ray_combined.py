import os
import logging
import numpy as np
import tensorflow as tf
from acme import specs, wrappers
from acme.tf import utils as tf2_utils
from omegaconf import DictConfig
from vnl_ray.agents.intention_network_factory import make_network_factory_dmpo_intention
from vnl_ray.tasks.mouse_reach import mouse_reach
from vnl_ray.agents.ray_distributed_dmpo import Learner
from vnl_ray.default_logger import make_default_logger
from vnl_ray.utils import plot_reward
from collections import defaultdict
import hydra
from vnl_ray.agents.utils_intention import get_rodent_egocentric_obs_key, get_mouse_egocentric_obs_key
from typing import Sequence, Callable, Any, List, OrderedDict
from dm_control import composer
import imageio


def load_policy_network(checkpoint_path, network_factory, action_spec):
    """
    Load a policy network from a checkpoint using the provided network factory.
    """
    print(f"Loading policy network from checkpoint: {checkpoint_path}")
    network_factory_result = network_factory(action_spec)

    # Ensure the network factory returns a dictionary
    if isinstance(network_factory_result, dict):
        policy_network = network_factory_result.get("policy")
    else:
        raise ValueError("Network factory did not return a dictionary.")

    # Initialize the network with dummy input
    dummy_obs = tf.zeros((1, 12), dtype=tf.float32)  # Explicit batch size of 1
    policy_network(dummy_obs)

    # Restore the checkpoint
    checkpoint = tf.train.Checkpoint(policy_network=policy_network)
    status = checkpoint.restore(checkpoint_path).expect_partial()
    print(f"Checkpoint restore status: {status}")
    return policy_network


def preprocess_observations(observations):
    """
    Preprocess observations by converting them to tensors and ensuring consistent batch dimensions.
    Handles dictionary-based observations and ensures compatibility with batch_concat.
    """
    if isinstance(observations, dict):
        # Convert all values to tensors and add a batch dimension if needed
        processed_observations = {}
        for key, value in observations.items():
            tensor = tf.convert_to_tensor(value)
            # Only add batch dimension if it doesn't already have one
            if len(tensor.shape) == 1 or (len(tensor.shape) > 1 and tensor.shape[0] != 1):
                tensor = tf2_utils.add_batch_dim(tensor)
            processed_observations[key] = tensor
        return tf2_utils.batch_concat(processed_observations)
    elif isinstance(observations, (tuple, list)):
        # Convert tuples or lists to tensors and add a batch dimension if needed
        tensors = []
        for value in observations:
            tensor = tf.convert_to_tensor(value)
            if len(tensor.shape) == 1 or (len(tensor.shape) > 1 and tensor.shape[0] != 1):
                tensor = tf2_utils.add_batch_dim(tensor)
            tensors.append(tensor)
        return tf.concat(tensors, axis=-1)
    else:
        # Convert single observations to tensors and add a batch dimension if needed
        tensor = tf.convert_to_tensor(observations)
        if len(tensor.shape) == 1 or (len(tensor.shape) > 1 and tensor.shape[0] != 1):
            return tf2_utils.add_batch_dim(tensor)
        return tensor


def render_with_rewards(env, policy, rollout_length=150):
    """
    Render the environment with rewards and collect frames.
    """
    frames = []
    rewards = defaultdict(list)
    timestep = env.reset()

    for _ in range(rollout_length):
        # Preprocess observations before passing to the policy
        obs = preprocess_observations(timestep.observation)
        # No need to add another batch dimension since preprocess_observations already takes care of it
        action = policy(obs).mean()
        timestep = env.step(action.numpy())
        frames.append(env.physics.render(camera_id=0))

        # Handle different reward types (dictionary or direct value)
        if isinstance(timestep.reward, dict):
            # If reward is a dictionary, iterate through its items
            for key, value in timestep.reward.items():
                rewards[key].append(value)
        else:
            # If reward is a direct value (array or scalar), store it under 'total'
            rewards["total"].append(float(timestep.reward))

        if timestep.last():
            break

    return frames, rewards


def get_mouse_task_obs_key() -> List[str]:
    """
    Returns the task observation keys for the mouse.
    """
    return [
        "mouse/to_target",
        "mouse/target_size",
    ]


def get_task_obs_size(obs_spec: OrderedDict, walker_type: str, visual_feature_size: int = 0) -> int:
    """Calculate the shape of the task specific observation sizes, based on the walker type"""
    if walker_type != "rodent" and walker_type != "mouse":
        raise ValueError(f"Walker type: {walker_type} did not implement yet. Currently supported rodent and mouse")

    egocentric_obs_key = get_mouse_task_obs_key()

    obs_shape = 0
    for i in set(obs_spec.keys()) - set(egocentric_obs_key):
        # iterate through all non-egocentric obs key
        shape = obs_spec[i].shape
        if shape == ():  # scalar
            obs_shape += 1
        else:
            obs_shape += shape[0]
    return obs_shape


tasks = {
    "mouse_reach": mouse_reach,
}


def save_video_and_log(frames, save_dir, filename):
    """
    Saves the rendered video and logs the directory.

    Args:
        frames (list): List of video frames.
        save_dir (str): Directory to save the video.
        filename (str): Name of the video file.
    """
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, filename)
    print(f"Saving {len(frames)} frames to {video_path}")
    with imageio.get_writer(video_path, fps=30) as video:
        for frame in frames:
            video.append_data(frame)
    print(f"Video saved to: {video_path}")


class PolicyWrapper:
    """
    A wrapper for policy networks that handles various return formats and gracefully
    collects activations when available.
    """

    def __init__(self, policy_network):
        self._policy_network = policy_network
        # Try to determine if this policy supports activations
        self._supports_activations = self._check_supports_activations()
        print(f"Created PolicyWrapper (supports_activations={self._supports_activations})")

    def _check_supports_activations(self):
        """Check if the policy supports returning activations by inspecting its code."""
        try:
            # Get the source code if available
            import inspect

            source = inspect.getsource(self._policy_network.__call__)
            return "return_activations" in source
        except:
            # If we can't inspect the source, assume it doesn't support activations
            return False

    def __call__(self, obs, return_activations=False):
        """
        Call the policy network with the given observations.

        Args:
            obs: Observations to pass to the policy
            return_activations: Whether to try to return activations

        Returns:
            If return_activations is True and supported: (action, activations)
            Otherwise: action
        """
        if return_activations and self._supports_activations:
            try:
                # Try calling with return_activations=True
                result = self._policy_network(obs, return_activations=True)

                # Handle different return formats
                if isinstance(result, tuple):
                    if len(result) == 3:  # (action, intention, activations)
                        action, intention, activations = result
                        # Extract mean from action distribution if needed
                        if hasattr(action, "mean") and callable(action.mean):
                            action = action.mean()
                        return action, activations, intention
                    elif len(result) == 2:  # (action, activations)
                        action, activations = result
                        # Extract mean from action distribution if needed
                        if hasattr(action, "mean") and callable(action.mean):
                            action = action.mean()
                        return action, activations

                # If result format is unexpected, fall back to action-only mode
                print("Unexpected result format from policy with return_activations=True")
                return self._get_action(obs), {}

            except Exception as e:
                print(f"Error getting activations: {e}")
                return self._get_action(obs), {}
        else:
            # Just get the action
            return self._get_action(obs)

    def _get_action(self, obs):
        """Get just the action from the policy network."""
        result = self._policy_network(obs)

        # Handle case where result is a distribution
        if hasattr(result, "mean") and callable(result.mean):
            return result.mean()
        return result


def render_with_rewards_and_data(env, policy, rollout_length=150):
    """
    Render the environment with rewards and collect frames, kinematics, activations, and intentions.

    Args:
        env (composer.Environment): The environment to simulate.
        policy (snt.Module): The policy network used to compute actions.
        rollout_length (int): Number of steps to render.

    Returns:
        tuple: Frames, rewards, kinematics, activations, and intentions.
    """
    # Create a robust policy wrapper
    policy_wrapper = PolicyWrapper(policy)

    frames = []
    rewards = defaultdict(list)
    kinematics = []
    activations = []
    intentions = []
    timestep = env.reset()

    for _ in range(rollout_length):
        obs = preprocess_observations(timestep.observation)

        # Get action and possibly activations
        result = policy_wrapper(obs, return_activations=True)

        if isinstance(result, tuple):
            if len(result) == 3:  # (action, activations, intention)
                action, activation, intention = result
                intentions.append(intention.numpy() if hasattr(intention, "numpy") else intention)
            else:  # (action, activations)
                action, activation = result
        else:
            action = result
            activation = {}

        activations.append(activation)

        # Convert action to numpy if needed
        action_np = action.numpy() if hasattr(action, "numpy") else action

        timestep = env.step(action_np)
        frames.append(env.physics.render(camera_id=0))

        # Collect rewards
        if isinstance(timestep.reward, dict):
            for key, value in timestep.reward.items():
                rewards[key].append(value)
        else:
            rewards["total"].append(float(timestep.reward))

        # Collect kinematics
        kinematics.append(
            {
                "qpos": env.physics.data.qpos.copy(),
                "qvel": env.physics.data.qvel.copy(),
            }
        )

        if timestep.last():
            break

    return frames, rewards, kinematics, activations, intentions


def verify_observation_order(env, observation_spec):
    """
    Verifies if the observation order matches the MouseEntity definition.

    Args:
        env (composer.Environment): The environment to simulate.
        observation_spec (OrderedDict): The observation spec from the environment.

    Returns:
        bool: True if the order matches, False otherwise.
    """
    mouse_obs_keys = [
        "mouse/joint_angles",  # Matches the "qpos" observable
        "mouse/joint_velocities",  # Matches the "qvel" observable
        "mouse/to_target",  # Custom observable for distance to target
        "mouse/target_size",  # Custom observable for target size
    ]
    env_obs_keys = list(observation_spec.keys())
    if mouse_obs_keys != env_obs_keys:
        print(f"Observation order mismatch! Expected: {mouse_obs_keys}, Found: {env_obs_keys}")
        return False
    print("Observation order matches MouseEntity definition.")
    return True


@hydra.main(version_base=None, config_path="./config", config_name="train_config_mouse_reach_offline_akira")
def main(config: DictConfig):
    print("CONFIG:", config)

    def environment_factory_mouse_reach() -> "composer.Environment":
        env = tasks["mouse_reach"](actuator_type=config.run_config.actuator_type)
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    # Set up environment
    env = mouse_reach(actuator_type=config.run_config.actuator_type, config=config)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)

    # Get specs
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()

    # Dummy environment and network for quick use, deleted later. # create this earlier to access the obs
    dummy_env = environment_factory_mouse_reach()

    # Create network factory
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
            config.learner_network["high_level_intention_size"] if config.learner_network["use_multi_decoder"] else None
        ),
    )

    # Load policy network
    policy_network = load_policy_network(
        checkpoint_path=config.learner_params.checkpoint_to_load,
        network_factory=network_factory,
        action_spec=action_spec,
    )

    # Verify observation order
    if not verify_observation_order(env, observation_spec):
        print("Observation order mismatch detected. Please check MouseEntity and environment setup.")

    # Print information about policy network for debugging
    print(f"Policy network type: {type(policy_network).__name__}")
    print(f"Policy callable: {callable(policy_network)}")
    if hasattr(policy_network, "__call__"):
        print(f"Policy __call__ method: {policy_network.__call__}")

    # Render and collect data with our improved policy wrapper
    frames, rewards, kinematics, activations, intentions = render_with_rewards_and_data(
        env, policy_network, rollout_length=config.env_params.rollout_length
    )

    # Save video
    save_video_and_log(frames, "/root/vast/eric/vnl-ray/videos", "mouse_reach_render.mp4")

    # Save kinematics, activations, and intentions
    np.save("/root/vast/eric/vnl-ray/data/kinematics.npy", kinematics)
    np.save("/root/vast/eric/vnl-ray/data/activations.npy", activations)
    np.save("/root/vast/eric/vnl-ray/data/intentions.npy", intentions)
    print("Kinematics, activations, and intentions saved.")

    # Debug rewards
    print(f"Collected rewards: {rewards}")


if __name__ == "__main__":
    main()
