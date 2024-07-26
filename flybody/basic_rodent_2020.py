# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Produces reference environments for rodent tasks."""

import functools

import numpy as np

from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import bowl
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.tasks import escape
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.tasks import reach

from dm_control.utils import io as resources
from dm_control.locomotion.tasks.reference_pose import types
import os
import h5py

# from flybody.fruitfly import rodent
from dm_control.locomotion.walkers import rodent

from flybody.tasks.rodent_imitation import WalkImitation
# from dm_control.locomotion.tasks.reference_pose import tracking

from flybody.tasks import tracking

from flybody.tasks.trajectory_loaders import (
    HDF5WalkingTrajectoryLoader,
    InferenceWalkingTrajectoryLoader,
)

_CONTROL_TIMESTEP = 0.02
_PHYSICS_TIMESTEP = 0.001
GHOST_OFFSET = np.array((0, 0, 0))


def rodent_escape_bowl(random_state=None):
    """Requires a rodent to climb out of a bowl-shaped terrain."""

    # Build a position-controlled rodent walker.
    walker = rodent.Rat(observable_options={"egocentric_camera": dict(enabled=True)})

    # Build a bowl-shaped arena.
    arena = bowl.Bowl(size=(20.0, 20.0), aesthetic="outdoor_natural")

    # Build a task that rewards the agent for being far from the origin.
    task = escape.Escape(
        walker=walker,
        arena=arena,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP,
    )

    return composer.Environment(
        time_limit=20,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def rodent_run_gaps(random_state=None, contact_termination=True): # enable contact termination
    """Requires a rodent to run down a corridor with gaps."""
    # Build a position-controlled rodent walker.
    walker = rodent.Rat(observable_options={"egocentric_camera": dict(enabled=True)})
  
    # Build a corridor-shaped arena with gaps, where the sizes of the gaps and
    # platforms are uniformly randomized.
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(0.4, 0.8),
        gap_length=distributions.Uniform(0.05, 0.2),
        corridor_width=2,
        corridor_length=40,
        aesthetic="outdoor_natural",
    )

    # Build a task that rewards the agent for running down the corridor at a
    # specific velocity.
    task = corr_tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(5, 0, 0),
      walker_spawn_rotation=0,
      target_velocity=1.0,
      contact_termination=contact_termination, 
      terminate_at_height=-0.3,
      physics_timestep=_PHYSICS_TIMESTEP,
      control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def rodent_maze_forage(random_state=None):
    """Requires a rodent to find all items in a maze."""

    # Build a position-controlled rodent walker.
    walker = rodent.Rat(observable_options={"egocentric_camera": dict(enabled=True)})

    # Build a maze with rooms and targets.
    wall_textures = labmaze_textures.WallTextures(style="style_01")
    arena = mazes.RandomMazeWithTargets(
        x_cells=11,
        y_cells=11,
        xy_scale=0.5,
        z_height=0.3,
        max_rooms=4,
        room_min_size=4,
        room_max_size=5,
        spawns_per_room=1,
        targets_per_room=3,
        wall_textures=wall_textures,
        aesthetic="outdoor_natural",
    )

    # Build a task that rewards the agent for obtaining targets.
    task = random_goal_maze.ManyGoalsMaze(
        walker=walker,
        maze_arena=arena,
        target_builder=functools.partial(
            target_sphere.TargetSphere,
            radius=0.05,
            height_above_ground=0.125,
            rgb1=(0, 0, 0.4),
            rgb2=(0, 0, 0.7),
        ),
        target_reward_scale=50.0,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP,
    )

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def rodent_two_touch(random_state=None):
    """Requires a rodent to tap an orb, wait an interval, and tap it again."""

    # Build a position-controlled rodent walker.
    walker = rodent.Rat(observable_options={"egocentric_camera": dict(enabled=True)})

    # Build an open floor arena
    arena = floors.Floor(size=(10.0, 10.0), aesthetic="outdoor_natural")

    # Build a task that rewards the walker for touching/reaching orbs with a
    # specific time interval between touches
    task = reach.TwoTouch(
        walker=walker,
        arena=arena,
        target_builders=[
            functools.partial(target_sphere.TargetSphereTwoTouch, radius=0.025),
        ],
        randomize_spawn_rotation=True,
        target_type_rewards=[25.0],
        shuffle_target_builders=False,
        target_area=(1.5, 1.5),
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP,
    )

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def walk_imitation(
    ref_path: str | None = None,
    random_state: np.random.RandomState | None = None,
    terminal_com_dist: float = 0.3,
):
    """
    Rodent walking imitation, following similar calling with fruitfly imitation
    """
    walker = rodent.Rat # pass callable
    arena = floors.Floor()

    current_directory = os.getcwd()
    TEST_FILE_PATH = os.path.join(current_directory, ref_path)

    with h5py.File(TEST_FILE_PATH, 'r') as f:
        dataset_keys = tuple(f.keys())
        dataset = types.ClipCollection(ids=dataset_keys,)

    # Set up the mocap tracking task
    task = tracking.MultiClipMocapTracking(
        walker=walker,
        arena=arena,
        ref_path=resources.GetResourceFilename(TEST_FILE_PATH),
        ref_steps=(1, 2, 3, 4, 5),
        min_steps=1,
        dataset=dataset,
        reward_type='comic',
        always_init_at_clip_start=True,
        ghost_offset=GHOST_OFFSET
    )
    time_limit = 10.0

    return composer.Environment(
        time_limit=time_limit,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )
