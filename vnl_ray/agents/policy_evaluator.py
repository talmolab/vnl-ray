"""Utilities for properly evaluating policies with full network reconstruction."""

import numpy as np
from acme import types
from acme.tf import utils as tf2_utils
import tensorflow as tf
import sonnet as snt
from vnl_ray.agents.utils_sonnet import Sequential
from acme.tf import networks as network_utils
from vnl_ray.agents import agent_dmpo
from vnl_ray.agents.utils_intention import separate_observation


def create_proper_policy_wrapper(snapshot_path, network_factory, environment_spec, debug=False):
    """
    Creates a properly reconstructed policy by:
    1. Loading the raw policy from snapshot
    2. Recreating the observation network
    3. Combining them in the correct Sequential structure

    Args:
        snapshot_path: Path to the policy snapshot
        network_factory: Function to create network components
        environment_spec: Environment specifications
        debug: Whether to print debug info

    Returns:
        A properly structured policy for evaluation
    """
    if debug:
        print(f"Loading snapshot from: {snapshot_path}")

    try:
        # Load the raw policy from snapshot
        raw_policy = tf.saved_model.load(snapshot_path)

        if debug:
            print(f"Successfully loaded policy of type: {type(raw_policy)}")

        # Recreate network structure using the network factory
        def wrapped_network_factory(action_spec):
            networks_dict = network_factory(action_spec)
            networks = agent_dmpo.DMPONetworks(
                policy_network=networks_dict.get("policy"),
                critic_network=networks_dict.get("critic"),
                observation_network=networks_dict.get("observation", tf.identity),
            )
            return networks

        # Create networks to get observation_network
        networks = wrapped_network_factory(environment_spec.actions)
        networks.init(environment_spec)

        if debug:
            print(f"Recreated network architecture")
            print(f"Observation network: {type(networks.observation_network)}")
            print(f"Policy network: {type(networks.policy_network)}")

        # Create a proper Sequential network for evaluation
        # This is crucial for the correct evaluation pipeline
        proper_policy = ActionClippingWrapper(
            Sequential(
                [
                    networks.observation_network,  # Step 1: Preprocess observations
                    raw_policy,  # Step 2: Apply raw policy
                    network_utils.StochasticMeanHead(),  # Step 3: Get mean action (for eval)
                ]
            ),
            environment_spec.actions,
        )

        if debug:
            print(f"Successfully created proper policy wrapper with action clipping")

        return proper_policy

    except Exception as e:
        print(f"Error in create_proper_policy_wrapper: {e}")
        # Return a minimal wrapper that will at least try to run
        return MinimalPolicyWrapper(raw_policy, environment_spec.actions)


class ActionClippingWrapper:
    """
    Wrapper that ensures actions are clipped to the environment's action space.
    This matches how actions are processed in standard ACME.
    """

    def __init__(self, policy, action_spec):
        self._policy = policy
        self._action_spec = action_spec
        print(
            f"ActionClippingWrapper initialized with action range: " f"[{action_spec.minimum}, {action_spec.maximum}]"
        )

    def __call__(self, observation, return_activations=False):
        if return_activations:
            action, activations = self._policy(observation, return_activations=True)
        else:
            action = self._policy(observation)
            activations = None

        # Convert to numpy if needed
        if hasattr(action, "numpy"):
            action = action.numpy()

        # Clip action to environment's action space
        clipped_action = np.clip(action, self._action_spec.minimum, self._action_spec.maximum)

        if return_activations:
            return clipped_action, activations
        return clipped_action


class MinimalPolicyWrapper:
    """Minimal policy wrapper as fallback with action clipping."""

    def __init__(self, policy, action_spec):
        self._policy = policy
        self._action_spec = action_spec
        print("WARNING: Using minimal policy wrapper as fallback with action clipping")

    def __call__(self, observation, return_activations=False):
        # Add batch dimension
        if isinstance(observation, dict):
            # For dictionary observations, try to handle them directly
            batched_dict = {k: tf2_utils.add_batch_dim(v) for k, v in observation.items()}
            try:
                # Try to use TF2 batch_concat first
                batched_observation = tf2_utils.batch_concat(batched_dict)
            except:
                # Fallback to manual concat if needed
                try:
                    concat_observation = tf.concat(
                        [tf.convert_to_tensor(v, dtype=tf.float32) for k, v in observation.items()], axis=-1
                    )
                    batched_observation = tf.expand_dims(concat_observation, 0)
                except:
                    # Last resort: try separate_observation from utils_intention
                    try:
                        batched_observation = separate_observation(batched_dict)
                    except:
                        raise ValueError("Could not process observation")
        else:
            batched_observation = tf2_utils.add_batch_dim(observation)

        # Call policy and extract action
        activations = None
        try:
            if return_activations:
                try:
                    distribution, activations = self._policy(batched_observation, return_activations=True)
                except:
                    distribution = self._policy(batched_observation)
            else:
                distribution = self._policy(batched_observation)

            # Try to get mean if available
            if hasattr(distribution, "mean") and callable(getattr(distribution, "mean")):
                action = distribution.mean()
            # Else try sampling
            elif hasattr(distribution, "sample") and callable(getattr(distribution, "sample")):
                action = distribution.sample()
            # If it's already a tensor, use it directly
            else:
                action = distribution

            # Remove batch dimension
            if hasattr(action, "shape") and len(action.shape) > 1 and action.shape[0] == 1:
                action = action[0]

            # Convert to numpy
            if hasattr(action, "numpy"):
                action = action.numpy()

            # Clip action to environment's action space
            action = np.clip(action, self._action_spec.minimum, self._action_spec.maximum)

            if return_activations:
                return action, (activations or {})
            return action

        except Exception as e:
            print(f"Error calling policy: {e}")
            # Return zeros as fallback
            zeros = np.zeros(self._action_spec.shape)
            if return_activations:
                return zeros, {}
            return zeros
