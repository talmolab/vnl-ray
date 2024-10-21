# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Environment wrapper which converts double-to-single precision. 
This also cast the uint to float
"""

from acme import specs
from acme import types
from acme.wrappers import base

import dm_env
import numpy as np
import tree


class SinglePrecisionWrapperFloat(base.EnvironmentWrapper):
    """Wrapper which converts environments from double- to single-precision."""

    def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        return timestep._replace(
            reward=_convert_value(timestep.reward),
            discount=_convert_value(timestep.discount),
            observation=_convert_value(timestep.observation),
        )

    def step(self, action) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.step(action))

    def reset(self) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.reset())

    def action_spec(self):
        return _convert_spec(self._environment.action_spec())

    def discount_spec(self):
        return _convert_spec(self._environment.discount_spec())

    def observation_spec(self):
        return _convert_spec(self._environment.observation_spec())

    def reward_spec(self):
        return _convert_spec(self._environment.reward_spec())


def _convert_spec(nested_spec: types.NestedSpec) -> types.NestedSpec:
    """Convert a nested spec."""

    def _convert_single_spec(spec: specs.Array):
        """Convert a single spec."""
        if np.issubdtype(spec.dtype, np.float64):
            dtype = np.float32
        elif np.issubdtype(spec.dtype, np.int64):
            dtype = np.int32
        elif np.issubdtype(spec.dtype, np.uint8):
            dtype = np.float32
        else:
            dtype = spec.dtype
        return spec.replace(dtype=dtype)

    return tree.map_structure(_convert_single_spec, nested_spec)


def _convert_value(nested_value: types.Nest) -> types.Nest:
    """Convert a nested value given a desired nested spec."""

    def _convert_single_value(value):
        if value is not None:
            value = np.array(value, copy=False)
            if np.issubdtype(value.dtype, np.float64):
                value = np.array(value, copy=False, dtype=np.float32)
            elif np.issubdtype(value.dtype, np.int64):
                value = np.array(value, copy=False, dtype=np.int32)
        return value

    return tree.map_structure(_convert_single_value, nested_value)


class RemoveVisionWrapper(base.EnvironmentWrapper):
    """Wrapper that removes the vision observables from the environment."""

    def __init__(self, environment: dm_env.Environment):
        super().__init__(environment)
        action_spec = environment.action_spec()
        action_spec.pop("walker/egocentric_camera")
        self._action_spec = action_spec

    def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        return timestep._replace(observation=timestep.observation.pop("walker/egocentric_camera"))

    def step(self, action) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.step(action))

    def reset(self) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.reset())

class RemoveObsWrapper:
    def __init__(self, environment: dm_env.Environment):
        '''Wrapper for only keeping qpos and qvel in the obs space'''
        super().__init__(environment)
        self.env = environment

    def reset(self) -> dm_env.TimeStep:
        time_step = self.env.reset()
        return self._filter_obs(time_step)

    def step(self, action) -> dm_env.TimeStep:
        time_step = self.env.step(action)
        return self._filter_obs(time_step)

    def _filter_obs(self, time_step):
        obs = time_step.observation
        # Assuming `qpos` and `qvel` are in the 'observations' dict of `obs`
        filtered_obs = {
            'qpos': obs.get('qpos', []),
            'qvel': obs.get('qvel', [])
        }
        return time_step._replace(observation=filtered_obs)