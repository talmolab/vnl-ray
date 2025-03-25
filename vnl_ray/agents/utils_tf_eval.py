"""Utilities for tensorflow evaluation with enhanced debugging and error handling."""

import numpy as np
import traceback
from acme import types
from acme.tf import utils as tf2_utils
import tensorflow as tf
from vnl_ray.agents.utils_intention import separate_observation


class TestPolicyEvalWrapper:
    """
    Special wrapper for evaluation that understands intention networks.
    """

    def __init__(self, policy, sample=False, debug=True):
        """
        Args:
            policy: Test policy, typically an intention network loaded from snapshot
            sample: Whether to return sample or mean of the distribution
            debug: Enable detailed debugging output
        """
        self._policy = policy
        self._sample = sample
        self._debug = debug

        # Check if the policy has intention_network attribute or is an intention network itself
        self._has_intention_network = hasattr(policy, "intention_network")

        print(f"TestPolicyEvalWrapper initialized with intention-aware handling")
        print(f"Policy type: {type(policy)}")
        print(f"Has intention_network attribute: {self._has_intention_network}")

        # Keep track of call counts for debugging
        self._call_counter = 0

        # Log detailed information about the policy structure if debug is on
        if self._debug:
            self._log_policy_details()

    def _log_policy_details(self):
        """Print detailed information about the policy structure."""
        print(f"\n===== POLICY DETAILS =====")
        print(f"Policy type: {type(self._policy)}")

        # Check for intention network
        if self._has_intention_network:
            print(f"  Has intention_network: {self._has_intention_network}")
            try:
                intention_network = self._policy.intention_network
                print(f"  Intention network type: {type(intention_network)}")

                # Check for encoder and decoder in intention network
                if hasattr(intention_network, "encoder"):
                    print(f"  Encoder type: {type(intention_network.encoder)}")
                if hasattr(intention_network, "decoder"):
                    print(f"  Decoder type: {type(intention_network.decoder)}")
            except Exception as e:
                print(f"  Error examining intention network: {e}")

        # Check for common attributes
        for attr in ["observation_network", "encoder", "decoder", "sample", "mean", "__call__"]:
            has_attr = hasattr(self._policy, attr)
            is_callable = has_attr and callable(getattr(self._policy, attr))
            print(f"  Has {attr}: {has_attr} (callable: {is_callable})")

        print("=========================\n")

    def __call__(self, observation: types.NestedArray, return_activations=False):
        """
        Enhanced policy wrapper with intention network awareness.

        Args:
            observation: Observation from the environment
            return_activations: Whether to return activation values

        Returns:
            action: The action to take
            activations: (Optional) activation values if return_activations=True
        """
        self._call_counter += 1
        if self._debug:
            print(f"\n----- Policy call #{self._call_counter} -----")
            print(f"Observation type: {type(observation)}")
            if isinstance(observation, dict):
                print(f"Observation keys: {list(observation.keys())}")
                print(f"Key shapes: {[(k, np.array(v).shape) for k, v in observation.items()]}")
            elif hasattr(observation, "shape"):
                print(f"Observation shape: {observation.shape}")

        # Process the observation based on type
        try:
            if isinstance(observation, dict):
                # Dictionary observation - use the special separate_observation function
                # which is specifically designed for intention networks
                if self._debug:
                    print("Using separate_observation for dictionary observation")
                batched_dict = {k: tf2_utils.add_batch_dim(v) for k, v in observation.items()}
                batched_observation = separate_observation(batched_dict)
                if self._debug:
                    print(f"Processed observation shape: {batched_observation.shape}")
            else:
                # Already a numpy array, just add batch dimension
                batched_observation = tf2_utils.add_batch_dim(observation)
                if self._debug:
                    print(f"Using array observation with shape {batched_observation.shape}")
        except Exception as e:
            print(f"Error processing observation: {e}")
            if self._debug:
                print(traceback.format_exc())
            # Return zero action as fallback
            action_shape = getattr(self, "_expected_action_size", 12)  # Default to 12 for mouse_reach
            if return_activations:
                return np.zeros(action_shape), {"error": str(e)}
            return np.zeros(action_shape)

        # Policy call with robust error handling
        try:
            if return_activations:
                try:
                    # For intention networks, we need to enable capturing activations
                    if self._debug:
                        print("Attempting policy call with return_activations=True")
                    distribution, activations = self._policy(batched_observation, return_activations=True)
                except Exception as e1:
                    if self._debug:
                        print(f"First attempt failed: {e1}")
                    # Fall back to regular call - we won't get activations
                    distribution = self._policy(batched_observation)
                    activations = {"note": "Activations not available - intention network error"}
            else:
                distribution = self._policy(batched_observation)
                activations = None
        except Exception as e:
            print(f"Policy call failed: {e}")
            if self._debug:
                print(traceback.format_exc())
            action_shape = getattr(self, "_expected_action_size", 12)
            if return_activations:
                return np.zeros(action_shape), {"error": str(e)}
            return np.zeros(action_shape)

        # Extract action from distribution
        try:
            # Check if it's a tensorflow probability distribution
            if hasattr(distribution, "mean") and callable(getattr(distribution, "mean")):
                if self._sample and hasattr(distribution, "sample") and callable(getattr(distribution, "sample")):
                    action = distribution.sample()
                else:
                    action = distribution.mean()
            else:
                # Just use the distribution as the action
                action = distribution

            # Convert to numpy if needed
            if hasattr(action, "numpy"):
                action = action.numpy()

            # Remove batch dimension if present
            if len(action.shape) > 1 and action.shape[0] == 1:
                action = action[0]

        except Exception as e:
            print(f"Error extracting action: {e}")
            if self._debug:
                print(traceback.format_exc())
            action_shape = getattr(self, "_expected_action_size", 12)
            action = np.zeros(action_shape)

        if self._debug:
            print(f"Final action shape: {action.shape}")

        # Return action and activations (if requested)
        if return_activations:
            if self._debug:
                print(f"Returning action and activations")
                if isinstance(activations, dict):
                    print(f"Activation keys: {list(activations.keys())}")
            return action, activations if activations is not None else {}

        return action
