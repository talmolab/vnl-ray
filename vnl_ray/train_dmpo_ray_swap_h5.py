"""
Script for distributed reinforcement learning training with Ray, loading policies from h5.

This script is based on train_dmpo_ray.py but modified to handle policies saved with
dictionary observation formats while working with environments that may use flat observations.
"""

# ruff: noqa: F821, E722, E402

# Start Ray cluster first, before imports.
import ray
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy,
    NodeAffinitySchedulingStrategy,
)
from ray.util.placement_group import placement_group
import logging
import os

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

import vnl_ray
from vnl_ray.agents.remote_as_local_wrapper import RemoteAsLocal
from vnl_ray.agents.counting import PicklableCounter
from vnl_ray.agents.network_factory import policy_loss_module_dmpo
from vnl_ray.agents.losses_mpo import PenalizationCostRealActions
from vnl_ray.agents.policy_format_adapter import FlatToDictPolicyWrapper
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

# Remove any pre-existing LD_LIBRARY_PATH and override it.
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ["LD_LIBRARY_PATH"] = "/root/miniforge3/envs/flybody/lib"

PYHTONPATH = os.path.dirname(os.path.dirname(vnl_ray.__file__))

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


# Function to wrap a policy if it's not already wrapped
def ensure_policy_wrapped(policy):
    """Ensure the policy is wrapped with FlatToDictPolicyWrapper if needed."""
    if hasattr(policy, "_policy"):  # Already a wrapper
        return policy
    print("Wrapping policy with FlatToDictPolicyWrapper")
    return FlatToDictPolicyWrapper(policy)


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="train_config_mouse_reach_akira",
)
def main(config: DictConfig) -> None:
    print("CONFIG:", config)

    # Add explicit check for the swap_decoder_with_jax parameter and ensure it's at top level
    if hasattr(config, "swap_decoder_with_jax"):
        print(f"swap_decoder_with_jax from config: {config.swap_decoder_with_jax}")
    elif hasattr(config.learner_params, "swap_decoder_with_jax"):
        # Check learner_params first (matches train_dmpo_ray.py structure)
        config.swap_decoder_with_jax = config.learner_params.swap_decoder_with_jax
        print(f"Moving swap_decoder_with_jax from learner_params to top level: {config.swap_decoder_with_jax}")
    elif hasattr(config.run_config, "swap_decoder_with_jax"):
        # Then check run_config as fallback
        config.swap_decoder_with_jax = config.run_config.swap_decoder_with_jax
        print(f"Moving swap_decoder_with_jax from run_config to top level: {config.swap_decoder_with_jax}")
    else:
        print("swap_decoder_with_jax not found in config, setting default to False")
        config.swap_decoder_with_jax = False

    # Print the exact config path being used for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Config path: {os.path.join(os.getcwd(), 'config', 'train_config_mouse_reach_akira.yaml')}")

    from vnl_ray.agents.ray_distributed_dmpo import (
        DMPOConfig,
        ReplayServer,
        Learner,
        EnvironmentLoop,
    )

    # Monkey patch the EnvironmentLoop class to wrap policies with FlatToDictPolicyWrapper
    original_environment_loop_init = EnvironmentLoop.__init__

    def patched_environment_loop_init(self, *args, **kwargs):
        original_environment_loop_init(self, *args, **kwargs)

        # Wrap the policy if this is an evaluator
        if hasattr(self, "_actor_or_evaluator") and self._actor_or_evaluator == "evaluator":
            if hasattr(self, "_actor") and hasattr(self._actor, "_policy"):
                print("Wrapping policy with FlatToDictPolicyWrapper in EnvironmentLoop init")
                self._actor._policy = FlatToDictPolicyWrapper(self._actor._policy)

    EnvironmentLoop.__init__ = patched_environment_loop_init

    # Monkey patch the load_snapshot_and_render method in a simpler way
    original_load_snapshot_and_render = EnvironmentLoop.load_snapshot_and_render

    def patched_load_snapshot_and_render(self, logging_data, **kwargs):
        # Keep a reference to the original policy
        original_policy = self._actor._policy

        # Replace policy with wrapped version for rendering
        self._actor._policy = ensure_policy_wrapped(original_policy)

        try:
            # Call the original method
            result = original_load_snapshot_and_render(self, logging_data, **kwargs)
            return result
        finally:
            # Always restore original policy even if there's an exception
            self._actor._policy = original_policy

    EnvironmentLoop.load_snapshot_and_render = patched_load_snapshot_and_render

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
        ),
        "mouse_reach": environment_factory_mouse_reach,
    }

    # Dummy environment and network for quick use, deleted later.
    dummy_env = environment_factories[config.run_config["task_name"]]()

    # Create network factory for RL task.
    if config.learner_network["use_intention"]:
        # Check if we should use JAX decoder instead of TensorFlow
        use_jax_decoder = config.get("swap_decoder_with_jax", config.run_config.get("swap_decoder_with_jax", False))

        # Create the base network factory
        network_factory = make_network_factory_dmpo_intention(
            task_obs_size=get_task_obs_size(
                dummy_env.observation_spec(),
                config.run_config["agent_name"],
                config.obs_network["visual_feature_size"],
            ),
            encoder_layer_sizes=config.learner_network["encoder_layer_sizes"],
            decoder_layer_sizes=config.learner_network["decoder_layer_sizes"],
            critic_layer_sizes=config.learner_network["critic_layer_sizes"],
            intention_size=config.learner_network["intention_size"],
            use_tfd_independent=True,
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

        # If JAX decoder is enabled, wrap the network factory to swap only the decoder
        if use_jax_decoder:
            print("Using JAX decoder instead of TensorFlow decoder")

            # Store the original factory
            original_factory = network_factory

            # Create a wrapper factory that swaps the decoder
            def jax_decoder_network_factory(action_spec):
                from vnl_ray.agents.intention_network_base import DecoderJAX

                # Get the original networks
                networks = original_factory(action_spec)
                policy_network = networks["policy"]

                # Only if the checkpoint is specified, swap the decoder
                if config.learner_params.get("checkpoint_to_load"):
                    print(f"Loading JAX decoder from checkpoint: {config.learner_params['checkpoint_to_load']}")

                    # Create JAX decoder with the same parameters
                    action_size = np.prod(action_spec.shape, dtype=int)
                    jax_decoder = DecoderJAX(
                        layer_sizes=config.learner_network["decoder_layer_sizes"],
                        layer_norm=True,
                        action_size=action_size,
                        min_scale=1e-6,
                    )

                    # Initialize the decoder (this calls the module once to build it)
                    dummy_input = tf.ones((1, policy_network.intention_size + 147))  # 147 is the egocentric size
                    jax_decoder(dummy_input)

                    # Load weights from checkpoint
                    jax_decoder.load_from_h5_checkpoint(config.learner_params["checkpoint_to_load"])

                    # Replace the decoder in the policy network
                    policy_network.decoder = jax_decoder
                    print("✓ JAX decoder successfully loaded and swapped")
                else:
                    print("Warning: No checkpoint specified, decoder not swapped")

                return networks

            # Replace the network factory with our wrapped version
            network_factory = jax_decoder_network_factory
    else:
        # online settings
        network_factory = make_network_factory_dmpo(
            action_spec=dummy_env.action_spec(),
            policy_layer_sizes=config.learner_network["policy_layer_sizes"],
            critic_layer_sizes=config.learner_network["critic_layer_sizes"],
        )

    dummy_net = network_factory(dummy_env.action_spec())
    environment_spec = specs.make_environment_spec(dummy_env)

    # This callable will be calculating penalization cost by converting canonical
    # actions to real (not wrapped) environment actions inside DMPO agent.
    penalization_cost = None

    # HARDCODED checkpoint directory to ensure correct path
    checkpoint_dir = "/root/vast/eric/vnl-ray/training/ray-mouse-mouse_reach-ckpts-h5/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Using hardcoded checkpoint directory: {checkpoint_dir}")

    # Create a custom logger function that properly captures swap_decoder_with_jax
    def custom_make_default_logger(*args, **kwargs):
        # Add swap_decoder_with_jax explicitly to logger config
        if "config" in kwargs and "userdata" in kwargs["config"]:
            if hasattr(config, "swap_decoder_with_jax"):
                kwargs["config"]["swap_decoder_with_jax"] = config.swap_decoder_with_jax
                kwargs["config"]["userdata"]["swap_decoder_with_jax"] = config.swap_decoder_with_jax
                kwargs["config"]["learner_params"]["swap_decoder_with_jax"] = config.swap_decoder_with_jax

        # Call the original logger
        return make_default_logger(*args, **kwargs)

    # Distributed DMPO agent configuration.
    dmpo_config = DMPOConfig(
        num_actors=config.env_params["num_actors"],
        batch_size=config.learner_params["batch_size"],
        discount=config.learner_params["discount"],
        prefetch_size=1024,
        num_learner_steps=1000,
        min_replay_size=50_000,
        max_replay_size=4_000_000,
        samples_per_insert=None,
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
        policy_optimizer=snt.optimizers.Adam(config.learner_params["policy_optimizer_lr"]),
        critic_optimizer=snt.optimizers.Adam(config.learner_params["critic_optimizer_lr"]),
        dual_optimizer=snt.optimizers.Adam(config.learner_params["dual_optimizer_lr"]),
        target_critic_update_period=107,
        target_policy_update_period=101,
        actor_update_period=5_000,
        log_every=30,
        logger=custom_make_default_logger,  # Use our custom logger wrapper
        logger_save_csv_data=False,
        checkpoint_max_to_keep=None,
        checkpoint_directory=checkpoint_dir,
        checkpoint_to_load=config.learner_params["checkpoint_to_load"],
        print_fn=None,
        userdata=dict(
            swap_decoder_with_jax=config.swap_decoder_with_jax
        ),  # Ensure it's in userdata right from the start
        kickstart_teacher_cps_path=(
            config.learner_params["kickstart_teacher_cps_path"]
            if "kickstart_teacher_cps_path" in config.learner_params
            else None
        ),
        kickstart_epsilon=(
            config.learner_params["kickstart_epsilon"] if "kickstart_epsilon" in config.learner_params else 0
        ),
        time_delta_minutes=5,
        eval_average_over=config.eval_params["eval_average_over"],
        KL_weights=(0, 0),
        load_decoder_only=(
            config.learner_params["load_decoder_only"] if "load_decoder_only" in config.learner_params else False
        ),
        froze_decoder=config.learner_params["froze_decoder"] if "froze_decoder" in config.learner_params else False,
    )

    # Print absolute checkpoint directory path for debugging
    print(f"Checkpoint directory (absolute): {os.path.abspath(dmpo_config.checkpoint_directory)}")

    dmpo_dict_config = dataclasses.asdict(dmpo_config)
    merged_config = dmpo_dict_config | OmegaConf.to_container(config)

    # Ensure swap_decoder_with_jax is in the merged_config at the top level
    merged_config["swap_decoder_with_jax"] = config.swap_decoder_with_jax

    # Remove any potential duplicate nested occurrences to avoid confusion
    if "run_config" in merged_config and "swap_decoder_with_jax" in merged_config["run_config"]:
        print(f"Removing duplicate swap_decoder_with_jax from run_config")
        del merged_config["run_config"]["swap_decoder_with_jax"]

    # Log for verification
    print(f"Final swap_decoder_with_jax value in merged_config: {merged_config['swap_decoder_with_jax']}")

    # Explicitly add it to userdata for the logger to capture (redundant but clearer)
    dmpo_config.userdata["swap_decoder_with_jax"] = merged_config["swap_decoder_with_jax"]

    logger_kwargs = {"config": merged_config}
    dmpo_config.userdata["logger_kwargs"] = logger_kwargs

    # Print to verify it's in the userdata and will be logged
    print(f"swap_decoder_with_jax in userdata: {dmpo_config.userdata['swap_decoder_with_jax']}")

    # Print full job config and full environment specs.
    print("\n", dmpo_config)
    print("\n", dummy_net)
    print("\nobservation_spec:\n", dummy_env.observation_spec())
    print("\naction_spec:\n", dummy_env.action_spec())
    print("\ndiscount_spec:\n", dummy_env.discount_spec())
    print("\nreward_spec:\n", dummy_env.reward_spec(), "\n")
    del dummy_env
    del dummy_net

    # Environment variables for learner, actor, and replay buffer processes.
    runtime_env_learner = {
        "env_vars": {
            "MUJOCO_GL": "osmesa",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "PYTHONPATH": PYHTONPATH,
            "LD_LIBRARY_PATH": "/root/miniforge3/envs/flybody/lib",
        }
    }
    runtime_env_actor = {
        "env_vars": {
            "MUJOCO_GL": "osmesa",
            "CUDA_VISIBLE_DEVICES": "-1",
            "PYTHONPATH": PYHTONPATH,
            "LD_LIBRARY_PATH": "/root/miniforge3/envs/flybody/lib",
        }
    }

    # Define resources and placement group
    gpu_pg = placement_group([{"GPU": 1, "CPU": 10}], strategy="STRICT_PACK")

    # === Create Replay Server.
    runtime_env_replay = {
        "env_vars": {
            "PYTHONPATH": PYHTONPATH,
            "LD_LIBRARY_PATH": "/root/miniforge3/envs/flybody/lib",
        }
    }

    ReplayServer = ray.remote(
        num_gpus=0,
        runtime_env=runtime_env_replay,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=gpu_pg),
    )(ReplayServer)

    replay_servers = dict()
    servers = []
    if "actors_envs" in config:
        dmpo_config.max_replay_size = dmpo_config.max_replay_size // 4
        for name, num_actors in config.actors_envs.items():
            if num_actors != 0:
                if not config["separate_replay_servers"]:
                    replay_server = ReplayServer.remote(dmpo_config, environment_spec)
                    addr = ray.get(replay_server.get_server_address.remote())
                    replay_server = RemoteAsLocal(replay_server)
                    servers.append(replay_server)
                    replay_servers["general"] = addr
                    print("SINGLE: Started Single replay server for this task.")
                    break
                elif "num_replay_servers" in config and config["num_replay_servers"] != 0:
                    dmpo_config.max_replay_size = dmpo_config.max_replay_size // config["num_replay_servers"]
                    for i in range(config["num_replay_servers"]):
                        _name = f"{name}-{i+1}"
                        replay_server = ReplayServer.remote(dmpo_config, environment_spec)
                        addr = ray.get(replay_server.get_server_address.remote())
                        print(f"MULTIPLE: Started Replay Server for task {_name} on {addr}")
                        replay_servers[_name] = addr
                        replay_server = RemoteAsLocal(replay_server)
                        servers.append(replay_server)
                        time.sleep(0.1)
                else:
                    replay_server = ReplayServer.remote(dmpo_config, environment_spec)
                    addr = ray.get(replay_server.get_server_address.remote())
                    print(f"MULTIPLE: Started Replay Server for task {name} on {addr}")
                    replay_servers[name] = addr
                    replay_server = RemoteAsLocal(replay_server)
                    servers.append(replay_server)
                    time.sleep(0.1)
    else:
        if "num_replay_servers" in config.env_params and config.env_params["num_replay_servers"] != 0:
            dmpo_config.max_replay_size = dmpo_config.max_replay_size // config.env_params["num_replay_servers"]
            for i in range(config.env_params["num_replay_servers"]):
                name = f"{config.run_config['task_name']}-{i+1}"
                replay_server = ReplayServer.remote(dmpo_config, environment_spec)
                addr = ray.get(replay_server.get_server_address.remote())
                print(f"MULTIPLE: Started Replay Server for task {name} on {addr}")
                replay_servers[name] = addr
                replay_server = RemoteAsLocal(replay_server)
                servers.append(replay_server)
                time.sleep(0.5)
        else:
            replay_server = ReplayServer.remote(dmpo_config, environment_spec)
            addr = ray.get(replay_server.get_server_address.remote())
            print(f"Started Replay Server on {addr}")
            replay_servers[config.run_config["task_name"]] = addr

    # === Create Counter.
    counter = ray.remote(PicklableCounter)
    counter = counter.remote()
    counter = RemoteAsLocal(counter)

    # === Create Learner.
    Learner = ray.remote(
        num_gpus=1,
        runtime_env=runtime_env_learner,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=gpu_pg),
    )(Learner)

    learner = Learner.remote(
        replay_servers,
        counter,
        environment_spec,
        dmpo_config,
        network_factory,
    )
    learner = RemoteAsLocal(learner)

    print("Waiting until learner is ready...")
    learner.isready(block=True)

    checkpointer_dir, snapshotter_dir = learner.get_checkpoint_dir()
    print("Checkpointer directory:", checkpointer_dir)
    print("Snapshotter directory:", snapshotter_dir)

    if not os.path.exists(checkpointer_dir):
        print(f"WARNING: Checkpoint directory {checkpointer_dir} does not exist!")
    elif not os.access(checkpointer_dir, os.W_OK):
        print(f"WARNING: Checkpoint directory {checkpointer_dir} is not writable!")
    else:
        print(f"Checkpoint directory {checkpointer_dir} exists and is writable ✓")

    # === Create Actors and Evaluator.
    EnvironmentLoop = ray.remote(num_gpus=0, runtime_env=runtime_env_actor)(EnvironmentLoop)

    n_actors = dmpo_config.num_actors

    def create_actors(n_actors, environment_factory, replay_server_addr):
        """Return list of requested number of actor instances."""
        actors = []
        for _ in range(n_actors):
            actor = EnvironmentLoop.remote(
                replay_server_address=replay_server_addr,
                variable_source=learner,
                counter=counter,
                network_factory=network_factory,
                environment_factory=environment_factory,
                dmpo_config=dmpo_config,
                actor_or_evaluator="actor",
            )
            actor = RemoteAsLocal(actor)
            actors.append(actor)
            time.sleep(0.01)
        return actors

    def create_evaluator(task_name, replay_server_addr):
        if task_name == "rodent_imitation":
            env_fact = functools.partial(environment_factories[task_name], random_range=0)
        else:
            env_fact = environment_factories[task_name]
        evaluator = EnvironmentLoop.remote(
            replay_server_address="",
            variable_source=learner,
            counter=counter,
            network_factory=network_factory,
            environment_factory=env_fact,
            dmpo_config=dmpo_config,
            actor_or_evaluator="evaluator",
            snapshotter_dir=snapshotter_dir,
            task_name=task_name,
        )
        return evaluator

    actors = []
    evaluators = []
    if "actors_envs" in config:
        print(config.actors_envs)
        for name, num_actors in config.actors_envs.items():
            if num_actors != 0:
                if not config.env_params["separate_replay_servers"]:
                    actors += create_actors(
                        num_actors,
                        environment_factories[name],
                        replay_servers["general"],
                    )
                elif "num_replay_servers" in config and config["num_replay_servers"] != 0:
                    for i in range(config["num_replay_servers"]):
                        _name = f"{name}-{i+1}"
                        num_actor_per_replay = num_actors // config["num_replay_servers"]
                        actors += create_actors(
                            num_actor_per_replay,
                            environment_factories[name],
                            replay_servers[_name],
                        )
                else:
                    actors += create_actors(num_actors, environment_factories[name], replay_servers[name])
                print(f"ACTOR Creation: {name}, has #{num_actors} of actors.")
            evaluators.append(RemoteAsLocal(create_evaluator(name, "")))
            print(f"EVALUTATOR Creation for task: {name}")
    else:
        print(f"ACTOR Creation: {n_actors}")
        if "num_replay_servers" in config.env_params:
            num_replay_server = config.env_params["num_replay_servers"]
        else:
            num_replay_server = 1
        for i in range(num_replay_server):
            name = f"{config.run_config['task_name']}-{i+1}"
            num_actor_per_replay = n_actors // num_replay_server
            actors += create_actors(
                num_actor_per_replay,
                environment_factories[config.run_config["task_name"]],
                replay_servers[name],
            )
        if config.run_config["task_name"] == "imitation_rodent":
            env_fac = functools.partial(
                environment_factories[config.run_config["task_name"]], always_init_at_clip_start=True
            )
        else:
            env_fac = environment_factories[config.run_config["task_name"]]
        evaluator = EnvironmentLoop.remote(
            replay_server_address="",
            variable_source=learner,
            counter=counter,
            network_factory=network_factory,
            environment_factory=env_fac,
            dmpo_config=dmpo_config,
            actor_or_evaluator="evaluator",
            snapshotter_dir=snapshotter_dir,
            task_name=config.run_config["task_name"],
        )
        evaluators.append(RemoteAsLocal(evaluator))

    print("Waiting until actors are ready...")
    for actor in actors:
        actor.isready(block=True)
    for evaluator in evaluators:
        evaluator.isready(block=True)

    print("Actors ready, issuing run command to all")

    # === Run all.
    if hasattr(counter, "run"):
        counter.run(block=False)
    for actor in actors:
        actor.run(block=False)
    for evaluator in evaluators:
        evaluator.run(block=False)

    while True:
        learner.run(block=True)


if __name__ == "__main__":
    main()
