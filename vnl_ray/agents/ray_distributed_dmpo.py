"""Classes for DMPO agent distributed with Ray."""

from typing import Iterator, Callable, List
import socket
import dataclasses
import copy
import logging
import os
import wandb
from pathlib import Path
import re
import imageio

import ray
import numpy as np

import tensorflow as tf
import reverb
import sonnet as snt

import acme
from acme import core
from acme import specs
from acme import datasets
from acme import adders
from acme import wrappers
from acme.utils import counting
from acme.utils import loggers
from acme.tf import variable_utils
from acme.tf import networks as network_utils
from acme.adders import reverb as reverb_adders

from vnl_ray.agents.learning_dmpo import DistributionalMPOLearner
from vnl_ray.agents import agent_dmpo
from vnl_ray.agents.actors import DelayedFeedForwardActor
from vnl_ray.utils import vision_rollout_and_render, rollout_and_render, render_with_rewards
from vnl_ray.agents.utils_tf import TestPolicyWrapper
from vnl_ray.agents.decoder_swap_utils import swap_decoder_with_jax

# logging & plotting
from matplotlib import pyplot as plt
from io import BytesIO

from vnl_ray.agents.utils_sonnet import Sequential


@dataclasses.dataclass
class DMPOConfig:
    num_actors: int = 32
    batch_size: int = 256
    prefetch_size: int = 4
    min_replay_size: int = 10_000
    max_replay_size: int = 4_000_000
    samples_per_insert: float = 32.0  # None: limiter = reverb.rate_limiters.MinSize()
    n_step: int = 5
    num_samples: int = 20
    num_learner_steps: int = 100
    clipping: bool = True
    discount: float = 0.95  # modify to align with Diego's mimic
    policy_loss_module: snt.Module | None = None
    policy_optimizer: snt.Optimizer | None = None
    critic_optimizer: snt.Optimizer | None = None
    dual_optimizer: snt.Optimizer | None = None
    target_policy_update_period: int = 101
    target_critic_update_period: int = 107
    actor_update_period: int = 1000
    logger: loggers.base.Logger | None = None
    log_every: float = 60.0  # Seconds.
    logger_save_csv_data: bool = False
    checkpoint_to_load: str | None = None  # Path to checkpoint.
    load_decoder_only: bool = False  # whether only loads decoder
    froze_decoder: bool = False  # whether we froze the weight of the decoder
    swap_decoder_with_jax: bool = False  # whether to swap decoder with JAX decoder
    decoder_h5_path: str = ""  # path to h5 file with decoder weights
    checkpoint_max_to_keep: int | None = 1  # None: keep all checkpoints.
    checkpoint_directory: str | None = "~/ray-ckpts/"  # None: no checkpointing.
    time_delta_minutes: float = 30
    terminal: str = "current_terminal"
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
    print_fn: Callable = logging.info
    userdata: dict | None = None
    actor_observation_callback: Callable | None = None
    config_dict: dict | None = None
    kickstart_teacher_cps_path: str = ("",)  # specify the location of the kickstarter teacher policy's cps
    kickstart_epsilon: float = (0.005,)
    eval_average_over: int = (200,)  # how many steps of statistic to average over in evaluator.
    KL_weights: List[float] = (0.0, 0.0)


class ReplayServer:
    """Reverb replay server, can be used with DMPO agent."""

    def __init__(self, config: DMPOConfig, environment_spec: specs.EnvironmentSpec):
        """Spawn a Reverb server with experience replay tables."""

        self._config = config
        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)
        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer,
            )

        replay_buffer = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=reverb_adders.NStepTransitionAdder.signature(environment_spec),
        )

        self._replay_server = reverb.Server(tables=[replay_buffer], port=None)
        # Get hostname and port of the server.
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        print("DEBUG: ", hostname, ip_address)
        port = self._replay_server.port
        self._replay_server_address = f"{ip_address}:{port}"

    def get_server_address(self):
        return self._replay_server_address

    def isready(self):
        """Dummy method to check if ReplayServer is ready."""
        pass


class Learner(DistributionalMPOLearner):
    """The Learning part of the DMPO agent."""

    def __init__(
        self,
        replay_server_addresses: dict,  # Allow multiple replay server address here. Modify corresponding logics.
        counter: counting.Counter,
        environment_spec: specs.EnvironmentSpec,
        dmpo_config,
        network_factory,
        label="learner",
    ):
        self._config = dmpo_config
        # self._reverb_client = reverb.Client(replay_server_address)
        self._reverb_clients = [reverb.Client(addr) for addr in replay_server_addresses.values()]
        self._label = label

        # Ensure checkpoint directory exists and is absolute
        if self._config.checkpoint_directory:
            # Force the correct path pattern
            if "/root/vast/vnl-ray/" in self._config.checkpoint_directory:
                self._config.checkpoint_directory = self._config.checkpoint_directory.replace(
                    "/root/vast/vnl-ray/", "/root/vast/eric/vnl-ray/"
                )
                print(f"Corrected checkpoint path to: {self._config.checkpoint_directory}")
            elif not self._config.checkpoint_directory.startswith("/root/vast/eric/vnl-ray/"):
                # Hardcode the path if it doesn't already have the correct prefix
                self._config.checkpoint_directory = "/root/vast/eric/vnl-ray/training/ray-mouse-mouse_reach-ckpts/"
                print(f"Reset to hardcoded checkpoint path: {self._config.checkpoint_directory}")

            os.makedirs(self._config.checkpoint_directory, exist_ok=True)
            print(f"Ensuring checkpoint directory exists: {self._config.checkpoint_directory}")

        def wrapped_network_factory(action_spec):
            networks_dict = network_factory(action_spec)
            networks = agent_dmpo.DMPONetworks(
                policy_network=networks_dict.get("policy"),
                critic_network=networks_dict.get("critic"),
                observation_network=networks_dict.get(
                    "observation", tf.identity
                ),  # optionally use the user defined observation network
                # if none is define, use the identity function.
            )
            return networks

        # Create the networks to optimize (online) and target networks.
        online_networks = wrapped_network_factory(environment_spec.actions)
        target_networks = copy.deepcopy(online_networks)
        # Initialize the networks.
        online_networks.init(environment_spec)
        target_networks.init(environment_spec)

        # Check if we should swap the decoder with a JAX decoder
        if dmpo_config.swap_decoder_with_jax and dmpo_config.decoder_h5_path:
            # This assumes the policy network has an 'intention_network' attribute
            # that contains the intention network with decoder
            if hasattr(online_networks.policy_network, "intention_network"):
                #                 print(f"\n======================================================")
                #                 print(f"SWAPPING DECODER: Replacing standard decoder with JAX decoder")
                #                 print(f"H5 CHECKPOINT PATH: {dmpo_config.decoder_h5_path}")
                #                 print(f"======================================================\n")

                decoder_layer_sizes = dmpo_config.userdata.get("decoder_layer_sizes", [512, 512, 512])
                if (
                    "learner_network" in dmpo_config.userdata.get("config", {})
                    and "decoder_layer_sizes" in dmpo_config.userdata["config"]["learner_network"]
                ):
                    decoder_layer_sizes = dmpo_config.userdata["config"]["learner_network"]["decoder_layer_sizes"]

                #                 print(f"Using decoder layer sizes: {decoder_layer_sizes}")

                swap_decoder_with_jax(
                    online_networks.policy_network.intention_network,
                    decoder_h5_path=dmpo_config.decoder_h5_path,
                    action_size=environment_spec.actions.shape[0],
                    layer_sizes=decoder_layer_sizes,
                    min_scale=0.1,
                    freeze_weights=dmpo_config.froze_decoder,
                )

                # Also swap the target network's decoder
                if hasattr(target_networks.policy_network, "intention_network"):
                    #                     print(f"Also swapping target network decoder...")
                    swap_decoder_with_jax(
                        target_networks.policy_network.intention_network,
                        decoder_h5_path=dmpo_config.decoder_h5_path,
                        action_size=environment_spec.actions.shape[0],
                        layer_sizes=decoder_layer_sizes,
                        min_scale=0.1,
                        freeze_weights=dmpo_config.froze_decoder,
                    )

        # print(f"\n✓ DECODER SWAP SUCCESSFULLY COMPLETED FOR ALL NETWORKS")
        # print(f"======================================================\n")

        datasets = [
            self._make_dataset_iterator(c) for c in self._reverb_clients
        ]  # (SY) add multiple reverbe client here
        counter = counting.Counter(parent=counter, prefix=label)
        if self._config.logger is None:
            logger = loggers.make_default_logger(
                label=label,
                time_delta=self._config.log_every,
                steps_key=f"{label}_steps",
                print_fn=self._config.print_fn,
                save_data=self._config.logger_save_csv_data,
            )
        else:
            if "logger_kwargs" in self._config.userdata:
                logger_kwargs = self._config.userdata["logger_kwargs"]
            else:
                logger_kwargs = {}
            logger = self._config.logger(
                label=label,
                time_delta=self._config.log_every,
                wandb_project=True,
                identity="learner",
                **logger_kwargs,
            )

        # Maybe checkpoint and snapshot the learner (saved in ~/acme/).
        checkpoint_enable = self._config.checkpoint_directory is not None
        if checkpoint_enable:
            print(f"Checkpointing enabled. Saving to: {self._config.checkpoint_directory}")
            print(f"Checkpoint interval: {self._config.time_delta_minutes} minutes")
        else:
            print("WARNING: Checkpointing is disabled!")

        # Have to call superclass constructor in this way.
        # Solved with Ray issue:  https://github.com/ray-project/ray/issues/449
        DistributionalMPOLearner.__init__(
            self,
            policy_network=online_networks.policy_network,
            critic_network=online_networks.critic_network,
            observation_network=online_networks.observation_network,
            target_policy_network=target_networks.policy_network,
            target_critic_network=target_networks.critic_network,
            target_observation_network=target_networks.observation_network,
            policy_loss_module=self._config.policy_loss_module,
            policy_optimizer=self._config.policy_optimizer,
            critic_optimizer=self._config.critic_optimizer,
            dual_optimizer=self._config.dual_optimizer,
            clipping=self._config.clipping,
            discount=self._config.discount,
            num_samples=self._config.num_samples,
            target_policy_update_period=self._config.target_policy_update_period,
            target_critic_update_period=self._config.target_critic_update_period,
            datasets=datasets,
            logger=logger,
            counter=counter,
            checkpoint_enable=checkpoint_enable,
            checkpoint_max_to_keep=self._config.checkpoint_max_to_keep,
            directory=self._config.checkpoint_directory,
            checkpoint_to_load=self._config.checkpoint_to_load,
            time_delta_minutes=self._config.time_delta_minutes,
            kickstart_teacher_cps_path=self._config.kickstart_teacher_cps_path,
            kickstart_epsilon=self._config.kickstart_epsilon,
            replay_server_addresses=replay_server_addresses,
            KL_weights=self._config.KL_weights,
            load_decoder_only=self._config.load_decoder_only,
            froze_decoder=self._config.froze_decoder,
        )

        # Verify checkpoint directories after initialization
        if self._checkpointer is not None:
            print(f"Checkpointer initialized with directory: {self._checkpointer._checkpoint_dir}")
            if not os.path.exists(self._checkpointer._checkpoint_dir):
                print(f"WARNING: Checkpointer directory does not exist: {self._checkpointer._checkpoint_dir}")
            elif not os.access(self._checkpointer._checkpoint_dir, os.W_OK):
                print(f"WARNING: Checkpointer directory not writable: {self._checkpointer._checkpoint_dir}")
            else:
                print(f"Checkpointer directory exists and is writable: {self._checkpointer._checkpoint_dir}")

    def _step(self, iterator):
        # Workaround to access _step in DistributionalMPOLearner:
        # @tf.function
        # def _step(self)
        #    ...
        return DistributionalMPOLearner._step(self, iterator)

    def run(self, num_steps=None):
        del num_steps  # Not used.
        # Run fixed number of learning steps and return control to have a chance
        # to process calls to `get_variables`.
        for _ in range(self._config.num_learner_steps):
            self.step()

    def isready(self):
        """Dummy method to check if learner is ready."""
        pass

    def get_checkpoint_dir(self):
        """Return Checkpointer and Snapshotter directories, if any."""
        if self._checkpointer is not None:
            checkpointer_dir = self._checkpointer._checkpoint_dir
            snapshotter_dir = self._snapshotter.directory if self._snapshotter else None
            print(f"Current checkpointer directory: {checkpointer_dir}")
            print(f"Current snapshotter directory: {snapshotter_dir}")
            return checkpointer_dir, snapshotter_dir
        print("WARNING: No checkpointer available!")
        return None, None

    def _make_dataset_iterator(
        self,
        reverb_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=reverb_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size,
        )
        return iter(dataset)


class EnvironmentLoop(acme.EnvironmentLoop):
    """Actor and Evaluator class."""

    def __init__(
        self,
        replay_server_address: str,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        network_factory,
        environment_factory,
        dmpo_config,
        actor_or_evaluator="actor",
        label=None,
        ray_head_node_ip: str | None = None,
        egl_device_id_head_node: list | None = None,
        egl_device_id_worker_node: list | None = None,
        task_name: str = "",
        snapshotter_dir: str | None = None,
    ):
        """The actor process."""

        # Maybe adjust EGL_DEVICE_ID environment variable internally in actor.
        if ray_head_node_ip is not None:
            current_node_id = ray.get_runtime_context().node_id.hex()
            running_on_head_node = False
            for node in ray.nodes():
                if node["NodeID"] == current_node_id and node["NodeManagerAddress"] == ray_head_node_ip:
                    running_on_head_node = True
                    break
            if running_on_head_node:
                egl_device_id = np.random.choice(egl_device_id_head_node)
            else:
                egl_device_id = np.random.choice(egl_device_id_worker_node)
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_id)

        assert actor_or_evaluator in ["actor", "evaluator"]
        self._actor_or_evaluator = actor_or_evaluator
        if actor_or_evaluator == "evaluator":
            del replay_server_address
        else:
            self._reverb_client = reverb.Client(replay_server_address)

        self._config = dmpo_config
        label = label or actor_or_evaluator

        # Create the environment.
        environment = environment_factory()
        environment_spec = specs.make_environment_spec(environment)

        def wrapped_network_factory(action_spec):
            networks_dict = network_factory(action_spec)
            networks = agent_dmpo.DMPONetworks(
                policy_network=networks_dict.get("policy"),
                critic_network=networks_dict.get("critic"),
                observation_network=networks_dict.get("observation", tf.identity),
            )
            return networks

        # Create the policy network, adder, ...
        networks = wrapped_network_factory(environment_spec.actions)
        networks.init(environment_spec)

        if actor_or_evaluator == "actor":
            # Actor: sample from policy_network distribution.
            policy_network = Sequential(
                [
                    networks.observation_network,
                    networks.policy_network,
                    network_utils.StochasticSamplingHead(),
                ]
            )
            adder = self._make_adder(self._reverb_client)
            save_data = False

        elif actor_or_evaluator == "evaluator":
            # Evaluator: get mean from policy_network distribution.
            policy_network = Sequential(
                [
                    networks.observation_network,
                    networks.policy_network,
                    network_utils.StochasticMeanHead(),
                ]
            )
            adder = None
            save_data = self._config.logger_save_csv_data

        # Create the agent.
        actor = self._make_actor(
            policy_network=policy_network,
            adder=adder,
            variable_source=variable_source,
            observation_callback=self._config.actor_observation_callback,
        )

        # Create logger and counter; actors will not spam bigtable.
        counter = counting.Counter(parent=counter, prefix=actor_or_evaluator)
        if self._config.logger is None:
            logger = loggers.make_default_logger(
                label=label,
                save_data=save_data,
                time_delta=self._config.log_every,
                steps_key=actor_or_evaluator + "_steps",
                print_fn=self._config.print_fn,
            )
        else:
            if "logger_kwargs" in self._config.userdata:
                logger_kwargs = self._config.userdata["logger_kwargs"]
            else:
                logger_kwargs = {}
            if actor_or_evaluator == "evaluator":
                print(f"Evaluator Node for Logger! Task Name: {task_name}")
            logger = self._config.logger(
                label=label,
                time_delta=self._config.log_every,
                # only create project for evaluators,
                wandb_project=actor_or_evaluator == "evaluator",
                identity="evaluator",
                task_name=task_name,
                **logger_kwargs,
            )

        if snapshotter_dir is not None:
            self._snapshotter_dir = Path(snapshotter_dir)
        self._latest_snapshot = None
        self._highest_snap_num = -1
        self._task_name = task_name
        self._environment_factory = environment_factory

        # for evaluator logging support for mean
        self._stats = []

        super().__init__(environment, actor, counter, logger)

    def run_episode(self) -> loggers.LoggingData:
        """Add rendering support for evaluator, and added aggregate stats"""
        logging_data = super().run_episode()
        # try:
        #     logging_data = super().run_episode()
        # except Exception as e:
        #     print(f"Exception: {e} encountered in run_episode. Returned Null result for this episode.")
        #     return {"episode_length": 0, "episode_return": 0, "steps_per_second": 0}  # TODO: This might causes error.
        if self._actor_or_evaluator == "evaluator":
            self._stats.append(logging_data)
            # self.stats is a [t1_data, t2_data, ...] array.
            if len(self._stats) >= self._config.eval_average_over:
                self._stats.pop(0)  # pop out the stats
            self.load_snapshot_and_render(logging_data)
            logging_data.update(self._eval_agg_stat(True))  # update in place
        return logging_data

    def _eval_agg_stat(self, include_raw=False) -> loggers.LoggingData:
        """
        For evaluators, calculates the aggregate statistics such as
        avg episode return, avg episode length, and avg sps

        _I am sorry, but this is some very hard to read list comprehension._
        """
        agg = {}
        stats_key = ["episode_length", "episode_return"]
        if len(self._stats) >= self._config.eval_average_over:  # only report summary statistic one a while
            avg = {
                f"avg_{key}": np.mean([d[key] for d in self._stats])
                for key in ["episode_length", "episode_return", "steps_per_second"]
            }
            var = {f"var_{key}": np.var([d[key] for d in self._stats]) for key in stats_key}
            maxi = {f"max_{key}": np.max([d[key] for d in self._stats]) for key in stats_key}
            mini = {f"min_{key}": np.min([d[key] for d in self._stats]) for key in stats_key}
            agg.update(avg)
            agg.update(var)
            agg.update(maxi)
            agg.update(mini)
            if include_raw:
                agg.update({f"curr_{key}": np.array([d[key] for d in self._stats]) for key in stats_key})
        return agg

    def load_snapshot_and_render(self, logging_data):
        """
        Check the snapshot directory, renders whenever there is a
        new policy snapshot, optionally send it to wandb. Modify the logging_data dict in place
        """
        render = False
        for path in self._snapshotter_dir.iterdir():
            match = re.match(r"policy-(\d+)", path.name)  # Look for the pattern "policy-number"
            if match:
                number = int(match.group(1))
                if number > self._highest_snap_num:
                    self._highest_snap_num = number
                    self._latest_snapshot = path
                    render = True
        if render:
            videos_path = self._snapshotter_dir.parent / "videos"
            videos_path.mkdir(parents=True, exist_ok=True)
            rendering_path = os.path.join(str(videos_path), f"{self._task_name}-{self._highest_snap_num}.mp4")
            # Frame width and height for rendering.
            render_kwargs = {"width": 640, "height": 480}
            try:
                policy = tf.saved_model.load(str(self._latest_snapshot))
                policy = TestPolicyWrapper(policy)
            except OSError as e:
                # sometime, the snapshotter will take a while to store the object. If the evaluator is
                print(f"Policy Loading Error: {e}. Skipping rendering for this policy.")
                self._highest_snap_num -= 1  # retry rendering the next iter.
                return
            # TODO: adapt the reward plotting to each task. Currently adapted: imitation/run-gaps
            env = self._environment_factory()
            env = wrappers.SinglePrecisionWrapper(env)
            env = wrappers.CanonicalSpecWrapper(env, clip=False)
            frames = render_with_rewards(env, policy, rollout_length=50 * 30)
            with imageio.get_writer(rendering_path, fps=1 / env.control_timestep()) as video:
                for f in frames:
                    video.append_data(f)
            logging_data["rollout"] = wandb.Video(rendering_path, format="mp4")

    def isready(self):
        """Dummy method to check if actor is ready."""
        pass

    def _make_actor(
        self,
        policy_network: snt.Module,
        adder: adders.Adder | None = None,
        variable_source: core.VariableSource | None = None,
        observation_callback: Callable | None = None,
    ):
        """Create an actor instance."""
        if variable_source:
            # Create the variable client responsible for keeping the actor up-to-date.
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={"policy": policy_network.variables},
                update_period=self._config.actor_update_period,  # was: hard-coded 1000,
            )
            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()
        else:
            variable_client = None

        # This is a modified version of actors.FeedForwardActor in Acme.
        return DelayedFeedForwardActor(
            policy_network=policy_network,
            adder=adder,
            variable_client=variable_client,
            action_delay=None,
            observation_callback=observation_callback,
        )

    def _make_adder(self, replay_client: reverb.Client) -> adders.Adder:
        """Create an adder which records data generated by the actor/environment."""
        return reverb_adders.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: lambda x: 1.0},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount,
        )
