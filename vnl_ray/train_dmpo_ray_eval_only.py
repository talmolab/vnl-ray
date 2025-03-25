"""
Script for distributed reinforcement learning training with Ray.

This script trains the fly-on-ball RL task using a distributed version of the
DMPO agent. The training runs in an infinite loop until terminated.

For lightweight testing, run this script with --test argument. It will run
training with a single actor and print training statistics every 10 seconds.

This script is not task-specific and can be used with other fly RL tasks by
swapping in other environments in the environment_factory function. The single
main configurable component below is the DMPO agent configuration and
training hyperparameters specified in the DMPOConfig data structure.
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
from pathlib import Path

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
from vnl_ray.agents import agent_dmpo  # Add this import for DMPONetworks
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


def render_and_log_rollout(snapshot_path, task_name, environment_factory, rollout_length=1500):
    """
    Render a rollout from the given snapshot and log it to WandB.
    """
    import imageio
    import wandb

    # Modified import to use TestPolicyWrapper and improved render_with_rewards.
    from vnl_ray.agents.ray_distributed_dmpo import TestPolicyWrapper, render_with_rewards

    print(f"Rendering rollout for snapshot: {snapshot_path}")

    # Create the environment
    env = environment_factory()
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=False)

    # Load the policy snapshot
    try:
        policy = tf.saved_model.load(snapshot_path)
        # Use the improved TestPolicyWrapper
        policy = TestPolicyWrapper(policy)
        print(f"Successfully loaded policy from {snapshot_path}")
    except Exception as e:
        print(f"Error loading policy snapshot: {e}")
        return None

    # Render the rollout
    videos_path = os.path.join(os.path.dirname(os.path.dirname(snapshot_path)), "videos")
    os.makedirs(videos_path, exist_ok=True)
    rendering_path = os.path.join(videos_path, f"{task_name}-manual-rollout.mp4")

    print(f"Rendering rollout with {rollout_length} frames...")
    frames = render_with_rewards(env, policy, rollout_length=rollout_length)

    if frames:
        print(f"Saving {len(frames)} frames to {rendering_path}")
        with imageio.get_writer(rendering_path, fps=1 / env.control_timestep()) as video:
            for frame in frames:
                video.append_data(frame)

        # Log the video to WandB
        try:
            wandb.init(project="mouse-reach-eval", name=f"{task_name}-manual-evaluation", resume=True)
            wandb.log({"manual_rollout": wandb.Video(rendering_path, format="mp4")})
            print("Video logged to WandB")
            wandb.finish()
        except Exception as e:
            print(f"Error logging to WandB: {e}")

        return rendering_path
    else:
        print("No frames were rendered!")
        return None


def safe_has_elements(arr):
    """
    Safely check if an array-like object has elements without triggering truth value ambiguity.

    Args:
        arr: Array-like object to check

    Returns:
        bool: True if the array has elements, False otherwise
    """
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


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="train_config_mouse_reach_akira",
)
def main(config: DictConfig) -> None:
    print("CONFIG:", config)

    from vnl_ray.agents.ray_distributed_dmpo import (
        DMPOConfig,
        ReplayServer,
        Learner,
        EnvironmentLoop,
    )

    print("\nRay context:")
    print(ray_context)

    ray_resources = ray.available_resources()
    print("\nAvailable Ray cluster resources:")
    print(ray_resources)

    # Create environment factory RL task.
    # Cannot parametrize it because it failed to serialize functions
    def environment_factory_mouse_reach() -> "composer.Environment":
        env = tasks["mouse_reach"](actuator_type=config.run_config.actuator_type, config=config)
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
            # termination_error_threshold=config["termination_error_threshold"], # TODO modify the config yaml for the imitation learning too.
        ),
        "mouse_reach": environment_factory_mouse_reach,
    }

    # Dummy environment and network for quick use, deleted later. # create this earlier to access the obs
    dummy_env = environment_factories[config.run_config["task_name"]]()

    # Create network factory for RL task. Config specify different ANN structures
    if config.learner_network["use_intention"]:
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
                config.learner_network["high_level_intention_size"]
                if config.learner_network["use_multi_decoder"]
                else None
            ),
        )
    else:
        # online settings
        network_factory = make_network_factory_dmpo(
            action_spec=dummy_env.action_spec(),
            policy_layer_sizes=config.learner_network["policy_layer_sizes"],
            critic_layer_sizes=config.learner_network["critic_layer_sizes"],
        )

    dummy_net = network_factory(dummy_env.action_spec())  # we should share this net for joint training
    # Get full environment specs.
    environment_spec = specs.make_environment_spec(dummy_env)

    # This callable will be calculating penalization cost by converting canonical
    # actions to real (not wrapped) environment actions inside DMPO agent.
    penalization_cost = None  # PenalizationCostRealActions(dummy_env.environment.action_spec())

    # HARDCODED checkpoint directory to ensure correct path
    checkpoint_dir = "/root/vast/eric/vnl-ray/training/ray-mouse-mouse_reach-ckpts/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Using hardcoded checkpoint directory: {checkpoint_dir}")

    # Distributed DMPO agent configuration.
    dmpo_config = DMPOConfig(
        num_actors=config.env_params["num_actors"],
        batch_size=config.learner_params["batch_size"],
        discount=config.learner_params["discount"],
        prefetch_size=1024,  # aggresive prefetch param, because we have large amount of data
        num_learner_steps=1000,
        min_replay_size=50_000,
        max_replay_size=4_000_000,
        samples_per_insert=None,  # allow less sample per insert to allow more data in # None is only min limiter
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
        policy_optimizer=snt.optimizers.Adam(config.learner_params["policy_optimizer_lr"]),  # reduce the lr
        critic_optimizer=snt.optimizers.Adam(config.learner_params["critic_optimizer_lr"]),
        dual_optimizer=snt.optimizers.Adam(config.learner_params["dual_optimizer_lr"]),
        target_critic_update_period=107,
        target_policy_update_period=101,
        actor_update_period=5_000,
        log_every=30,
        logger=make_default_logger,
        logger_save_csv_data=False,
        checkpoint_max_to_keep=None,
        checkpoint_directory=checkpoint_dir,
        checkpoint_to_load=config.learner_params["checkpoint_to_load"],
        print_fn=None,  # print # this causes issue pprint does not work
        userdata=dict(),
        kickstart_teacher_cps_path=(
            config.learner_params["kickstart_teacher_cps_path"]
            if "kickstart_teacher_cps_path" in config.learner_params
            else None
        ),  # specify the location of the kickstarter teacher policy's cps
        kickstart_epsilon=(
            config.learner_params["kickstart_epsilon"] if "kickstart_epsilon" in config.learner_params else 0
        ),
        time_delta_minutes=5,
        eval_average_over=config.eval_params["eval_average_over"],
        KL_weights=(0, 0),  # Keep KL regularization for intention space only
        # specify the KL with intention & action output layer # do not penalize the output layer # disabled it for now.
        load_decoder_only=(
            config.learner_params["load_decoder_only"] if "load_decoder_only" in config.learner_params else False
        ),
        froze_decoder=config.learner_params["froze_decoder"] if "froze_decoder" in config.learner_params else False,
    )

    # Print absolute checkpoint directory path for debugging
    print(f"Checkpoint directory (absolute): {os.path.abspath(dmpo_config.checkpoint_directory)}")

    dmpo_dict_config = dataclasses.asdict(dmpo_config)
    merged_config = dmpo_dict_config | OmegaConf.to_container(config)  # merged two config

    logger_kwargs = {"config": merged_config}
    dmpo_config.userdata["logger_kwargs"] = logger_kwargs

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
            "LD_LIBRARY_PATH": "/root/miniforge3/envs/flybody/lib",  # explicit new path
        }
    }
    runtime_env_actor = {
        "env_vars": {
            "MUJOCO_GL": "osmesa",
            "CUDA_VISIBLE_DEVICES": "-1",  # CPU-actors don't use CUDA.
            "PYTHONPATH": PYHTONPATH,
            "LD_LIBRARY_PATH": "/root/miniforge3/envs/flybody/lib",  # explicit new path
        }
    }

    # Define resources and placement group
    gpu_pg = placement_group([{"GPU": 1, "CPU": 10}], strategy="STRICT_PACK")

    # === Create Replay Server.
    runtime_env_replay = {
        "env_vars": {
            "PYTHONPATH": PYHTONPATH,  # Also used for counter.
            "LD_LIBRARY_PATH": "/root/miniforge3/envs/flybody/lib",  # add explicit new path
        }
    }

    ReplayServer = ray.remote(
        num_gpus=0,
        runtime_env=runtime_env_replay,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=gpu_pg),  # test out performance w/o
    )(ReplayServer)

    replay_servers = dict()  # {task_name: addr} # TODO: Probably could simplify this logic quite a bit
    servers = []
    if "actors_envs" in config:
        dmpo_config.max_replay_size = dmpo_config.max_replay_size // 4  # reduce each replay buffer size by 4.
        for name, num_actors in config.actors_envs.items():
            if num_actors != 0:
                if not config["separate_replay_servers"]:
                    # mixed experience replay buffers
                    replay_server = ReplayServer.remote(
                        dmpo_config, environment_spec
                    )  # each envs will share the same environment spec and dmpo_config
                    addr = ray.get(replay_server.get_server_address.remote())
                    replay_server = RemoteAsLocal(replay_server)
                    servers.append(replay_server)
                    replay_servers["general"] = addr
                    print("SINGLE: Started Single replay server for this task.")
                    break
                elif "num_replay_servers" in config and config["num_replay_servers"] != 0:
                    dmpo_config.max_replay_size = (
                        dmpo_config.max_replay_size // config["num_replay_servers"]
                    )  # shrink down the replay size correspondingly
                    for i in range(config["num_replay_servers"]):
                        _name = f"{name}-{i+1}"
                        # multiple replay server for load balancing
                        replay_server = ReplayServer.remote(
                            dmpo_config, environment_spec
                        )  # each envs will share the same environment spec and dmpo_config
                        addr = ray.get(replay_server.get_server_address.remote())
                        print(f"MULTIPLE: Started Replay Server for task {_name} on {addr}")
                        replay_servers[_name] = addr
                        replay_server = RemoteAsLocal(replay_server)
                        # this line is essential to keep a refernce to the replay server
                        # otherwise the object will be garbage collected and clean out
                        servers.append(replay_server)
                        time.sleep(0.1)
                        # multiple replay server setup
                else:
                    replay_server = ReplayServer.remote(
                        dmpo_config, environment_spec
                    )  # each envs will share the same environment spec and dmpo_config
                    addr = ray.get(replay_server.get_server_address.remote())
                    print(f"MULTIPLE: Started Replay Server for task {name} on {addr}")
                    replay_servers[name] = addr
                    replay_server = RemoteAsLocal(replay_server)
                    # this line is essential to keep a refernce to the replay server
                    # otherwise the object will be garbage collected and clean out
                    servers.append(replay_server)
                    time.sleep(0.1)
    else:
        if "num_replay_servers" in config.env_params and config.env_params["num_replay_servers"] != 0:
            dmpo_config.max_replay_size = (
                dmpo_config.max_replay_size // config.env_params["num_replay_servers"]
            )  # shrink down the replay size correspondingly
            for i in range(config.env_params["num_replay_servers"]):
                name = f"{config.run_config['task_name']}-{i+1}"
                # multiple replay server for load balancing
                replay_server = ReplayServer.remote(
                    dmpo_config, environment_spec
                )  # each envs will share the same environment spec and dmpo_config
                addr = ray.get(replay_server.get_server_address.remote())
                print(f"MULTIPLE: Started Replay Server for task {name} on {addr}")
                replay_servers[name] = addr
                replay_server = RemoteAsLocal(replay_server)
                # this line is essential to keep a refernce to the replay server
                # otherwise the object will be garbage collected and clean out
                servers.append(replay_server)
                time.sleep(0.5)
        else:
            # single replay server
            replay_server = ReplayServer.remote(dmpo_config, environment_spec)
            addr = ray.get(replay_server.get_server_address.remote())
            print(f"Started Replay Server on {addr}")
            replay_servers[config.run_config["task_name"]] = addr

    # === Create Counter.
    counter = ray.remote(PicklableCounter)  # This is class (direct call to ray.remote decorator).
    counter = counter.remote()  # Instantiate.
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
    # Verify directory exists and is writable
    if not os.path.exists(checkpointer_dir):
        print(f"WARNING: Checkpoint directory {checkpointer_dir} does not exist!")
    elif not os.access(checkpointer_dir, os.W_OK):
        print(f"WARNING: Checkpoint directory {checkpointer_dir} is not writable!")
    else:
        print(f"Checkpoint directory {checkpointer_dir} exists and is writable ✓")

    # === Create Actors and Evaluator.

    EnvironmentLoop = ray.remote(num_gpus=0, runtime_env=runtime_env_actor)(EnvironmentLoop)

    n_actors = dmpo_config.num_actors

    def create_actors(n_actors, environment_factory, replay_server_addr):  # callalbe env factoryory
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

    # IMPORTANT: In eval-only mode, only create training actors, NOT evaluators yet
    actors = []
    if "actors_envs" in config:
        # if the config file specifies diverse actor envs
        # created for multi-task RL
        print(config.actors_envs)
        for name, num_actors in config.actors_envs.items():
            if num_actors != 0:
                if not config.env_params["separate_replay_servers"]:
                    actors += create_actors(
                        num_actors,
                        environment_factories[name],
                        replay_servers["general"],
                    )  # mixed experience replay buffer
                elif "num_replay_servers" in config and config["num_replay_servers"] != 0:
                    # Handle multiple replay servers for load balancing
                    num_replay_servers = config["num_replay_servers"]
                    num_actor_per_replay = num_actors // num_replay_servers
                    for i in range(num_replay_servers):
                        _name = f"{name}-{i+1}"
                        actors += create_actors(
                            num_actor_per_replay,
                            environment_factories[name],
                            replay_servers[_name],
                        )
                        print(f"ACTOR Creation: {_name}, has #{num_actor_per_replay} actors.")
                else:
                    actors += create_actors(num_actors, environment_factories[name], replay_servers[name])
                    print(f"ACTOR Creation: {name}, has #{num_actors} actors.")
    else:
        # Get actors for a single task
        print(f"ACTOR Creation: {n_actors}")
        if "num_replay_servers" in config.env_params:
            num_replay_servers = config.env_params["num_replay_servers"]
        else:
            num_replay_servers = 1
        num_actor_per_replay = n_actors // num_replay_servers
        for i in range(num_replay_servers):
            name = f"{config.run_config['task_name']}-{i+1}"
            actors += create_actors(
                num_actor_per_replay,
                environment_factories[config.run_config["task_name"]],
                replay_servers[name],
            )
            print(f"ACTOR Creation: {name}, has #{num_actor_per_replay} actors.")

    print("Waiting until actors are ready...")
    # Block until all actors are ready
    for actor in actors:
        actor.isready(block=True)

    # Since we don't create any evaluators yet (only actors), proceed directly to working with the checkpoint
    print("Actors ready, preparing evaluator with proper snapshot path...")

    # Create snapshot from checkpoint if it doesn't exist
    checkpoint_path = config.learner_params["checkpoint_to_load"]

    # Extract checkpoint number from the path (handle both ckpt-NNN and checkpoint_NNN formats)
    if "ckpt-" in checkpoint_path:
        checkpoint_num = checkpoint_path.split("ckpt-")[-1]
    elif "checkpoint_" in checkpoint_path:
        checkpoint_num = checkpoint_path.split("checkpoint_")[-1]
    else:
        # Try to extract any number at the end of the filename
        checkpoint_num = os.path.basename(checkpoint_path).split("-")[-1]

    # Extract the run directory by finding the path segment before 'checkpoints'
    path_parts = checkpoint_path.split("/")
    try:
        checkpoints_index = path_parts.index("checkpoints")
        # Get everything up to but not including "checkpoints"
        run_dir = "/".join(path_parts[:checkpoints_index])
    except ValueError:
        # Fallback if "checkpoints" isn't in the path
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))

    # Construct the snapshot path
    checkpoint_snapshot_dir = os.path.join(run_dir, "snapshots")
    os.makedirs(checkpoint_snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(checkpoint_snapshot_dir, f"policy-{checkpoint_num}")

    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Extracted checkpoint number: {checkpoint_num}")
    print(f"Run directory: {run_dir}")
    print(f"Creating snapshot directory at: {checkpoint_snapshot_dir}")
    print(f"Snapshot path will be: {snapshot_path}")

    # Check if a policy snapshot already exists, generate one if not
    if not os.path.exists(snapshot_path):
        print(f"Policy snapshot doesn't exist at {snapshot_path}, will create automatically")
        # The learner will automatically create a snapshot when evaluating

    # Convert to Path object to ensure consistent handling
    checkpoint_snapshot_dir_path = Path(checkpoint_snapshot_dir)

    # Use string representation consistently
    snapshot_path_str = str(os.path.join(checkpoint_snapshot_dir, f"policy-{checkpoint_num}"))

    # Debug print to verify the directory exists
    print(f"Verifying checkpoint snapshot directory: {checkpoint_snapshot_dir_path}")
    print(f"Directory exists: {os.path.exists(checkpoint_snapshot_dir_path)}")
    print(f"Directory is readable: {os.access(checkpoint_snapshot_dir_path, os.R_OK)}")
    print(f"Snapshot path exists: {os.path.exists(snapshot_path_str)}")
    print(f"Snapshot path is readable: {os.access(snapshot_path_str, os.R_OK)}")

    # Verify snapshot_path before passing to remote function
    if not os.path.exists(snapshot_path_str):
        # Create empty snapshot directory if it doesn't exist
        print(f"WARNING: Snapshot path {snapshot_path_str} doesn't exist. Creating empty directory...")
        os.makedirs(os.path.dirname(snapshot_path_str), exist_ok=True)
        # We won't be able to render without a valid snapshot

    # Additional debug logging
    print(f"FINAL SNAPSHOT PATH BEING PASSED: {snapshot_path_str}")

    # Import the policy evaluator
    from vnl_ray.agents.policy_evaluator import create_proper_policy_wrapper

    # Later in the code where you're setting up the evaluator:
    proper_policy = create_proper_policy_wrapper(
        snapshot_path=snapshot_path_str, network_factory=network_factory, environment_spec=environment_spec, debug=True
    )

    # Use this properly wrapped policy for evaluation with EnvironmentLoop
    evaluator = EnvironmentLoop.options(name="explicit_evaluator").remote(
        replay_server_address="",
        variable_source=learner,
        counter=counter,
        network_factory=network_factory,
        environment_factory=env_fac,
        dmpo_config=dmpo_config,
        actor_or_evaluator="evaluator",
        snapshotter_dir=snapshotter_dir,  # Learner's snapshot directory
        checkpoint_snapshot_dir=str(checkpoint_snapshot_dir_path),
        task_name=config.run_config["task_name"],
        force_render=True,  # Force render in eval-only mode
        snapshot_path=snapshot_path_str,  # Pass as string to avoid serialization issues
        restored_policy=proper_policy,  # Pass the properly wrapped policy
    )
    evaluator = RemoteAsLocal(evaluator)

    # Wait for initialization and verify the parameters were set correctly
    print("Waiting for evaluator to initialize...")
    evaluator.isready(block=True)

    try:
        # RemoteAsLocal wrapper makes remote calls look like local calls
        path_verified = evaluator.verify_snapshot_path(snapshot_path_str)
        print(f"Evaluator snapshot path verification: {path_verified}")
    except Exception as e:
        print(f"Error verifying snapshot path: {e}")
        path_verified = False

    # === Run only what's needed for evaluation
    if hasattr(counter, "run"):
        counter.run(block=False)

    # Run the evaluator only if path was verified
    if path_verified:
        print("Running evaluator...")
        evaluator.run(block=False)
    else:
        print("ERROR: Failed to verify snapshot path. Evaluator may not render correctly.")

    try:
        # Keep the script running to allow evaluators to complete episodes
        # Periodically print status updates
        eval_runtime = 10 * 60  # Run for 10 minutes by default
        start_time = time.time()

        print(f"Evaluation will run for {eval_runtime/60:.1f} minutes")

        while time.time() - start_time < eval_runtime:
            elapsed = time.time() - start_time
            remaining = eval_runtime - elapsed
            print(f"Evaluation in progress - elapsed: {elapsed:.1f}s, remaining: {remaining:.1f}s")

            # Sleep for a bit but not too long, so Ctrl+C works reasonably
            time.sleep(30)

        print("Evaluation time completed. Waiting for evaluators to finish current episodes...")

        # Wait for evaluators to finish current episodes
        evaluator.isready(block=True)

        print("All evaluators have completed. Exiting.")
    except KeyboardInterrupt:
        print("Evaluation stopped by user")


if __name__ == "__main__":
    main()
