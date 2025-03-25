"""Utilities for tensorflow networks and nested data structures."""

import numpy as np
from acme import types
from acme.tf import utils as tf2_utils
import tensorflow as tf


class TestPolicyWrapper:
    """At test time, wraps policy to work with non-batched observations.
    Works with distributional policies, e.g. trained with the DMPO agent."""

    def __init__(self, policy, sample=False, observation_network=None):
        """
        Args:
            policy: Test policy, e.g. trained policy loaded as
                policy = tf.saved_model.load('path/to/snapshot').
            sample: Whether to return sample or mean of the distribution.
            observation_network: Optional preprocessing network to apply before policy.
        """
        self._policy = policy
        self._sample = sample
        self._observation_network = observation_network
        print(f"TestPolicyWrapper initialized. Policy already has obs network: {observation_network is not None}")

    def __call__(self, observation: types.NestedArray, return_activations=False):
        # Add a dummy batch dimension and as a side effect convert numpy to TF,
        # batched_observation: types.NestedTensor.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # Only apply observation network if provided AND observation isn't already a dictionary
        # This prevents applying it twice in normal training
        if self._observation_network is not None and not isinstance(observation, dict):
            try:
                batched_observation = self._observation_network(batched_observation)
                print("Applied observation network to non-dict observation")
            except Exception as e:
                print(f"Error applying observation network: {e}")

        # First try the standard approach - pass the batched observation directly
        try:
            if return_activations:
                distribution, activations = self._policy(batched_observation, return_activations=True)
            else:
                distribution = self._policy(batched_observation)
                activations = None
        except (TypeError, ValueError) as e:
            print(f"Basic policy call failed: {e}")
            # If original call fails and we're dealing with a dictionary observation,
            # try converting to a flat tensor (for evaluation with saved models that expect tensors)
            if isinstance(observation, dict):
                try:
                    print("Converting dictionary observation to tensor")
                    # Sort keys to ensure consistent order
                    concat_observation = tf.concat(
                        [
                            tf.convert_to_tensor(observation[key], dtype=tf.float32)
                            for key in sorted(observation.keys())
                        ],
                        axis=-1,
                    )
                    # Add batch dimension
                    concat_batched = tf.expand_dims(concat_observation, 0)

                    if return_activations:
                        distribution, activations = self._policy(concat_batched, return_activations=True)
                    else:
                        distribution = self._policy(concat_batched)
                        activations = None
                except Exception as inner_e:
                    print(f"Tensor conversion approach also failed: {inner_e}")
                    raise ValueError(f"Could not process observation with policy: {e} and then {inner_e}")
            else:
                # If it's not a dict observation, re-raise the original error
                raise

        # Get either the sample or mean from the distribution
        try:
            if self._sample:
                action = distribution.sample()
            else:
                action = distribution.mean()

            # Remove batch dimension
            action = action[0].numpy() if len(action.shape) > 1 else action.numpy()
        except Exception as e:
            print(f"Error extracting action: {e}")
            # Fall back to directly using the distribution if it's not a proper distribution object
            action = (
                distribution[0].numpy()
                if hasattr(distribution, "shape") and len(distribution.shape) > 1
                else distribution.numpy()
            )

        # Return action and activations (if requested)
        if return_activations:
            return action, activations
        else:
            return action
