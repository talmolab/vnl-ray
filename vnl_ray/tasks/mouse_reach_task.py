from dm_control import composer
from dm_control.composer import variation
from dm_control.utils import rewards
import numpy as np
import collections
import random
from dm_control.suite.utils import randomizers


class MouseReachTask(composer.Task):
    """
    A task where a mouse (loaded from an XML file) reaches for a target.

    This task sets up a mouse to perform reaching movements to a randomized
    target position. The task rewards the mouse for reaching the target and penalizes
    for disallowed contacts or failing to stay upright.

    Attributes:
        _target_size: Float representing the size of the target.
        _target_pos: List of available target positions.
        _randomize_pose: Boolean indicating whether to randomize the initial mouse pose.
        _randomize_target: Boolean indicating whether to randomize the target position.
        _failure_termination: Boolean indicating whether the episode should be terminated.
    """

    def __init__(
        self,
        mouse,
        arena,
        target_list="old_targets",
        randomize_pose=False,
        randomize_target=False,
        contact_termination=False,
        physics_timestep=0.001,
        control_timestep=0.001,
        config=None,
    ):
        """
        Initializes the MouseReachTask.

        Args:
            mouse: An instance of a custom mouse entity loaded from an XML file.
            arena: An instance of an environment's arena (the space where the agent operates).
            target_size: Size of the target.
            target_positions: Optional list of predefined target positions.
            randomize_pose: Boolean, whether to randomize the initial pose.
            randomize_target: Boolean, whether to randomize the target position.
            contact_termination: Whether to terminate the episode upon invalid contact.
            physics_timestep: Physics simulation timestep.
            control_timestep: Control input timestep.
        """
        self._arena = arena
        self._mouse = mouse
        self._randomize_pose = randomize_pose
        self._randomize_target = randomize_target
        self._contact_termination = contact_termination
        self._target_list = target_list
        self._config = config

        self._reward_keys = ["distance_reward"]

        # Get margin from config, default to 0.01 if not provided
        self._margin = 0.008  # self._config.run_config["margin"]

        # Attach the mouse to the arena
        self._arena.attach(self._mouse)

        # Enable default observables
        enabled_observables = self._mouse.observables

        # Set physics and control timesteps
        self.set_timesteps(physics_timestep=physics_timestep, control_timestep=control_timestep)

        self.last_reward_channels = []  # Initialize the last_reward_channels attribute

        self._target_sizes = [0.001, 0.002, 0.003]

    @property
    def root_entity(self):
        """
        Defines the root entity of the task.

        Returns:
            The arena instance that serves as the root entity of the task.
        """
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        """
        Initializes the MJCF model for the episode.

        Args:
            random_state: A NumPy random state instance for consistent randomization.
        """
        self._arena.regenerate(random_state)
        self._arena.mjcf_model.visual.map.znear = 0.00025
        self._arena.mjcf_model.visual.map.zfar = 4.0

    def initialize_episode(self, physics, random_state):
        """
        Sets the initial state of the environment at the start of each episode.

        Args:
            physics: A Physics object used for simulating the environment.
            random_state: A NumPy random state instance for consistent randomization.
        """
        # Reinitialize the mouse's pose
        self._mouse.reinitialize_pose(physics, random_state)
        self._mouse.disable_contacts(physics)

        self._target_size = random.choice(self._target_sizes)

        # Set the target size in the physics model using the MouseEntity's method
        self._mouse.set_target_size(physics, self._target_size)
        shoulder_pos = physics.model.joint("mouse/sh_elv").pos
        fingertip_pos = physics.named.data.geom_xpos["mouse/finger_tip"]

        # GENERATE TARGET LISTS
        if self._target_list == "old_targets":
            # adjusted_targets = [
            #     [0.0025355, 0.012, -0.0024645],  # previous runs: [0.0025355, 0.012, -0.0024645]
            # ]
            adjusted_targets = [
                [0.004, 0.012, -0.006],
                [0.0025355, 0.012, -0.0024645],
                [-0.001, 0.012, -0.001],
                [-0.0045355, 0.012, -0.0024645],
                [-0.006, 0.012, -0.006],
                [-0.0045355, 0.012, -0.0095355],
                [-0.001, 0.012, -0.011],
                [0.0025355, 0.012, -0.0095355],
            ]
        elif self._target_list == "new_targets":
            # Compute the differences in x and y between fingertip and shoulder.
            dx = fingertip_pos[0] - shoulder_pos[0]
            dy = fingertip_pos[1] - shoulder_pos[1]

            # Compute the radius in the x-y plane.
            radius_xy = np.sqrt(dx**2 + dy**2)

            # Compute the angle between shoulder and fingertip in the x-y plane.
            angle_ft = np.arctan2(dy, dx)

            # Number of points per row.
            N = 4

            # Set the target angle (adjust this if needed).
            target_angle = np.pi  # 180 degrees

            # Compute the angular difference between angle_ft and target_angle.
            total_span = np.arctan2(np.sin(target_angle - angle_ft), np.cos(target_angle - angle_ft))

            # Generate angles from angle_ft towards target_angle.
            angles = angle_ft + np.linspace(0, total_span, N)

            # Use the shoulder position as the center.
            x_center = shoulder_pos[0]
            y_center = shoulder_pos[1]

            # Compute the x and y coordinates for the target points.
            x_points = x_center + radius_xy * np.cos(angles)
            y_points = y_center + radius_xy * np.sin(angles)

            # Define two z positions, z1 and z2 (0.003 units apart).
            z1 = fingertip_pos[2]
            z2 = z1 + 0.003

            # Create two rows of z positions.
            z_points_row1 = np.full(N, z1)
            z_points_row2 = np.full(N, z2)

            # Combine x, y, z coordinates into two arrays for each row.
            points_row1 = np.vstack((x_points, y_points, z_points_row1)).T
            points_row2 = np.vstack((x_points, y_points, z_points_row2)).T

            # Combine both rows into a single array (8 points total).
            circle_points = np.vstack((points_row1, points_row2))

            # Find the smallest sphere radius.
            r_smallest = min(self._target_sizes)

            # Adjust target positions based on sphere sizes and apply an x-axis shift.
            adjusted_targets = []
            x_offset = 0.002  # Shift   targets along x-axis so they are in front of the arm.
            for point in circle_points:
                v = point - fingertip_pos
                v_mag = np.linalg.norm(v)
                v_hat = v / v_mag  # Unit vector.
                delta = self._target_size - r_smallest
                adjusted_center = point + delta * v_hat
                # Apply the x offset to shift the target.
                adjusted_center[0] += x_offset
                adjusted_targets.append(adjusted_center)

        # Randomly select one of the adjusted target points.
        target_index = random_state.randint(len(adjusted_targets))
        selected_target = adjusted_targets[target_index]

        # Set the target position in the physics model.
        physics.named.model.geom_pos["mouse/target", "x"] = selected_target[0]
        physics.named.model.geom_pos["mouse/target", "y"] = selected_target[1]
        physics.named.model.geom_pos["mouse/target", "z"] = selected_target[2]

        # Reset the failure termination condition.
        self._failure_termination = False

        return selected_target

    def _is_disallowed_contact(self, contact):
        """
        Determines whether a disallowed contact has occurred between the mouse and the arena.

        Args:
            contact: A contact object describing the interaction between two geoms.

        Returns:
            Boolean indicating whether the contact is disallowed.
        """
        set1, set2 = self._mouse_nonfoot_geomids, self._ground_geomids
        return (contact.geom1 in set1 and contact.geom2 in set2) or (contact.geom1 in set2 and contact.geom2 in set1)

    def before_step(self, physics, action, random_state):
        """
        Applies actions to the mouse before the simulation step. Actuator limits are set to [-1, 1].

        Args:
            physics: A Physics object used for simulating the environment.
            action: The control input to be applied to the mouse.
            random_state: A NumPy random state instance for consistent randomization.
        """
        # clipped_action = np.clip(action, -1.0, 1.0)
        physics.data.ncon = 0  # disable contacts, joint limits will serve as constraints
        self._mouse.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state):
        """
        Checks for any conditions after a simulation step, such as failure termination.

        Args:
            physics: A Physics object used for simulating the environment.
            random_state: A NumPy random state instance for consistent randomization.
        """
        self._failure_termination = False
        if self._contact_termination:
            for c in physics.data.contact:
                if self._is_disallowed_contact(c):
                    self._failure_termination = True
                    break

    def get_reward(self, physics):
        """
        Calculates and returns the reward for the current step based on the distance
        between the mouse's finger tip and the target.

        Args:
            physics: A Physics object used for simulating the environment.

        Returns:
            Float representing the calculated reward based on the proximity to the target.
        """
        # Ensure physics has the necessary attribute.
        if not hasattr(physics.named.data, "geom_xpos"):
            raise ValueError("Invalid physics object: missing geom_xpos attribute")
        # Compute the distance between target and finger tip.
        finger_to_target_dist = np.linalg.norm(
            physics.named.data.geom_xpos["mouse/target"] - physics.named.data.geom_xpos["mouse/finger_tip"]
        )
        # Use the target size and the configured margin (set in __init__) for tolerance.
        reward = rewards.tolerance(finger_to_target_dist, bounds=(0, self._target_size), margin=self._margin)
        self.last_reward_channels = {"distance_reward": reward}
        return reward

    def should_terminate_episode(self, physics):
        """
        Returns whether the episode should be terminated due to failure conditions.

        Args:
            physics: A Physics object used for simulating the environment.

        Returns:
            Boolean indicating whether the episode should terminate.
        """
        return self._failure_termination

    def get_discount(self, physics):
        """
        Returns the discount factor for the current episode step, indicating whether
        the episode is continuing or terminating.

        Args:
            physics: A Physics object used for simulating the environment.

        Returns:
            Float representing the discount factor. Returns 0 if the episode should terminate,
            otherwise returns 1.
        """
        return 0.0 if self._failure_termination else 1.0
