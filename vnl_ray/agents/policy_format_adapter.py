"""Policy wrapper for handling format conversions between flat and dictionary observations."""

import tensorflow as tf
import numpy as np
import sonnet as snt
from typing import Dict, Any, Union, List


class FlatToDictPolicyWrapper:
    """Wraps a policy that expects dictionary observations to accept flat observations."""

    def __init__(self, policy, obs_structure=None):
        """
        Args:
            policy: A policy that expects dictionary observations.
            obs_structure: A dictionary specifying how to split the flat observation.
                Example: {
                    'mouse/joint_angles': 4,
                    'mouse/joint_velocities': 4,
                    'mouse/to_target': 3,
                    'mouse/target_size': 1
                }
                If None, will use default mouse_reach observation structure.
        """
        self._policy = policy
        self._obs_structure = obs_structure or {
            "mouse/joint_angles": 4,
            "mouse/joint_velocities": 4,
            "mouse/to_target": 3,
            "mouse/target_size": 1,
        }
        print(f"FlatToDictPolicyWrapper initialized with structure: {self._obs_structure}")

    def __call__(self, observation):
        """Convert flat observation to dictionary and call the wrapped policy."""
        if isinstance(observation, dict):
            # Already in dictionary format, pass through
            return self._policy(observation)

        # Convert flat observation to dictionary
        start_idx = 0
        obs_dict = {}

        for key, size in self._obs_structure.items():
            obs_dict[key] = observation[..., start_idx : start_idx + size]
            start_idx += size

        print(f"Converted flat observation {observation.shape} to dictionary with keys: {list(obs_dict.keys())}")

        return self._policy(obs_dict)


class DictToFlatPolicyWrapper:
    """Wraps a policy that expects flat observations to accept dictionary observations."""

    def __init__(self, policy, keys_order=None):
        """
        Args:
            policy: A policy that expects flat observations.
            keys_order: The order of keys to concatenate the dictionary values.
        """
        self._policy = policy
        self._keys_order = keys_order
        print("DictToFlatPolicyWrapper initialized.")

    def __call__(self, observation):
        """Convert dictionary observation to flat and call the wrapped policy."""
        if not isinstance(observation, dict):
            # Already flat, pass through
            return self._policy(observation)

        # Convert dictionary to flat observation
        if self._keys_order is None:
            # Just concatenate all values in dictionary
            flat_obs = tf.concat([observation[k] for k in sorted(observation.keys())], axis=-1)
        else:
            # Concatenate in specified order
            flat_obs = tf.concat([observation[k] for k in self._keys_order], axis=-1)

        return self._policy(flat_obs)
