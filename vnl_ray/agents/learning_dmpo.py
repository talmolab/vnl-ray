"""Distributional MPO learner implementation."""

import time
from typing import List, Optional
import re

import acme
from acme import types
from acme.tf import losses
from acme.tf import networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from vnl_ray.utils import vision_rollout_and_render
from vnl_ray.agents.intention_network_base import IntentionNetwork
from vnl_ray.agents.utils_sonnet import Sequential

# inject our wandb logger for learner only


class DistributionalMPOLearner(acme.Learner):
    """Distributional MPO learner."""

    def __init__(
        self,
        policy_network: snt.Module,
        critic_network: snt.Module,
        target_policy_network: snt.Module,
        target_critic_network: snt.Module,
        discount: float,
        num_samples: int,
        target_policy_update_period: int,
        target_critic_update_period: int,
        datasets: List[tf.data.Dataset],
        observation_network: types.TensorTransformation = tf.identity,
        target_observation_network: types.TensorTransformation = tf.identity,
        policy_loss_module: Optional[snt.Module] = None,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        dual_optimizer: Optional[snt.Optimizer] = None,
        clipping: bool = True,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint_enable: bool = True,
        checkpoint_max_to_keep: Optional[int] = 1,  # If None, all checkpoints are kept.
        directory: str | None = "~/acme/",
        checkpoint_to_load: Optional[str] = None,
        load_decoder_only: bool = False,  # whether we only load the decoder from the previous checkpoint, but not other network.
        froze_decoder: bool = False,  # whether we want to froze the weight of the decoder
        time_delta_minutes: float = 15.0,
        kickstart_teacher_cps_path: str = None,  # specify the location of the kickstarter teacher policy's cps
        kickstart_epsilon: float = 0.005,
        replay_server_addresses: dict = None,
        KL_weights: List[float] = (0, 0),
    ):
        """
        KL_weights: list of float that specify the KL regularizer strength for the intention and action layer
        """

        # Store online and target networks.
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network

        # Make sure observation networks are snt.Module's so they have variables.
        self._observation_network = tf2_utils.to_sonnet_module(observation_network)
        self._target_observation_network = tf2_utils.to_sonnet_module(target_observation_network)

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("learner")

        # Other learner parameters.
        self._discount = discount
        self._num_samples = num_samples
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_policy_update_period = target_policy_update_period
        self._target_critic_update_period = target_critic_update_period

        # Batch dataset and create iterator.
        # Created multiple iterators for different replaybuffer sources
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterators = [iter(dataset) for dataset in datasets]  # pytype: disable=wrong-arg-types

        self._policy_loss_module = policy_loss_module or losses.MPO(
            epsilon=1e-1,
            epsilon_penalty=1e-3,
            epsilon_mean=1e-3,
            epsilon_stddev=1e-6,
            init_log_temperature=1.0,
            init_log_alpha_mean=1.0,
            init_log_alpha_stddev=10.0,
        )

        # Create the optimizers.
        self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
        self._dual_optimizer = dual_optimizer or snt.optimizers.Adam(1e-2)

        # Load the teacher's policy
        self._kickstart_teacher_policy = None
        self._kickstart_epsilon = 0
        if kickstart_teacher_cps_path != None:
            print("KICKSTART: Loading Teacher Policy from: {kickstart_teacher_cps_path}")
            self._kickstart_teacher_policy = tf.saved_model.load(kickstart_teacher_cps_path)
            self._kickstart_epsilon = tf.constant(kickstart_epsilon)

        # KL regularizing related
        self._KL_weights = KL_weights
        self._KL_regularized = max(self._KL_weights) != 0
        if self._KL_regularized and isinstance(self._target_policy_network, IntentionNetwork):
            # create standard normal distribution for KL divergence calculation.
            self._intention_std_normal_dist = tfd.MultivariateNormalDiag(
                loc=tf.zeros(self._target_policy_network.intention_size),
                scale_diag=tf.ones(self._target_policy_network.intention_size),
            )
            self._action_std_normkal_dist = tfd.MultivariateNormalDiag(
                loc=tf.zeros(self._target_policy_network.action_size),
                scale_diag=tf.ones(self._target_policy_network.action_size),
            )

            # if we want to use tdf independent (event size mismatch now.)
            # self._intention_std_normal_dist = tfd.Independent(
            #     tfd.Normal(
            #         loc=tf.zeros(self._target_policy_network.intention_size),
            #         scale=tf.ones(self._target_policy_network.intention_size),
            #     )
            # )
            # self._action_std_normkal_dist = tfd.Independent(
            #     tfd.Normal(
            #         loc=tf.zeros(self._target_policy_network.action_size),
            #         scale=tf.ones(self._target_policy_network.action_size),
            #     )
            # )

        self._replay_server_addresses = replay_server_addresses

        # Expose the variables.
        policy_network_to_expose = Sequential([self._target_observation_network, self._target_policy_network])
        self._variables = {
            "critic": self._target_critic_network.variables,
            "policy": policy_network_to_expose.variables,
        }

        # Create a checkpointer and snapshotter object.
        self._checkpointer = None
        self._snapshotter = None

        if checkpoint_enable:
            objects_to_save = {
                "counter": self._counter,
                "policy": self._policy_network,
                "critic": self._critic_network,
                "observation": self._observation_network,
                "target_policy": self._target_policy_network,
                "target_critic": self._target_critic_network,
                "target_observation": self._target_observation_network,
                "policy_optimizer": self._policy_optimizer,
                "critic_optimizer": self._critic_optimizer,
                "dual_optimizer": self._dual_optimizer,
                "policy_loss_module": self._policy_loss_module,
                "num_steps": self._num_steps,
            }
            if isinstance(self._target_policy_network, IntentionNetwork):
                # objects_to_save["policy_decoder"] = self._target_policy_network.high_level_encoder
                if self._target_policy_network.use_multi_encoder:
                    objects_to_save["policy_high_level_encoder"] = self._target_policy_network.high_level_encoder
                    objects_to_save["policy_mid_level_encoder"] = self._target_policy_network.mid_level_encoder
                else:
                    objects_to_save["policy_encoder"] = self._target_policy_network.encoder
            self._checkpointer = tf2_savers.Checkpointer(
                subdirectory="dmpo_learner",
                objects_to_save=objects_to_save,
                directory=directory,
                time_delta_minutes=time_delta_minutes,
                max_to_keep=checkpoint_max_to_keep,
            )

            objects_to_save = {
                "policy-0": Sequential([self._target_observation_network, self._target_policy_network]),
                # "policy-only-no-obs-network-0": snt.Sequential([self._target_policy_network]), '
                # we don't need to do kickstarting for now
            }
            # optionally save the decoder if we have one.
            if isinstance(self._target_policy_network, IntentionNetwork):
                objects_to_save["policy-decoder-0"] = self._target_policy_network.decoder
                if self._target_policy_network.use_multi_encoder:
                    objects_to_save["policy_high_level_encoder-0"] = self._target_policy_network.high_level_encoder
                    objects_to_save["policy_mid_level_encoder-0"] = self._target_policy_network.mid_level_encoder
                else:
                    objects_to_save["policy_encoder-0"] = self._target_policy_network.encoder

            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save=objects_to_save,
                directory=directory,
                time_delta_minutes=time_delta_minutes,
            )

        # Maybe load checkpoint.
        print(checkpoint_to_load)
        if checkpoint_to_load is not None and not load_decoder_only:
            _checkpoint = tf.train.Checkpoint(
                counter=tf2_savers.SaveableAdapter(self._counter),
                policy=self._policy_network,
                critic=self._critic_network,
                observation=self._observation_network,
                target_policy=self._target_policy_network,
                target_critic=self._target_critic_network,
                target_observation=self._target_observation_network,
                policy_optimizer=self._policy_optimizer,
                critic_optimizer=self._critic_optimizer,
                dual_optimizer=self._dual_optimizer,
                policy_loss_module=self._policy_loss_module,
                num_steps=self._num_steps,
            )
            status = _checkpoint.restore(checkpoint_to_load)  # noqa: F841
            print(f"CKPTS: LOADED checkpoint from {checkpoint_to_load}")
            # The assert below will not work because at this point not all variables have
            # been created in tf.train.Checkpoint argument objects. However, it's good
            # enough to revive a job from its checkpoint. Another option is to put
            # the checkpoint loading in the self.step method below, then the assertion
            # will work.
            # status.assert_existing_objects_matched()  # Sanity check.

        if checkpoint_to_load is not None and load_decoder_only:
            # Q: do we want to fix the online policy decoder too?
            _decoder_checkpoint = tf.train.Checkpoint(policy_decoder=self._target_policy_network.decoder)
            status = _decoder_checkpoint.restore(checkpoint_to_load)
            print(f"CKPTS: Decoder LOADED checkpoint from {checkpoint_to_load}")
        if froze_decoder:
            self._target_policy_network.decoder.trainable = False  # freeze the weights of decoder
            print(f"CKPTS: Decoder weight frozen.")

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    @tf.function
    def _step(self, iterator) -> types.NestedTensor:
        # Update target network.
        online_policy_variables = self._policy_network.variables
        target_policy_variables = self._target_policy_network.variables
        online_critic_variables = (
            *self._observation_network.variables,
            *self._critic_network.variables,
        )
        target_critic_variables = (
            *self._target_observation_network.variables,
            *self._target_critic_network.variables,
        )

        # Make online policy -> target policy network update ops.
        if tf.math.mod(self._num_steps, self._target_policy_update_period) == 0:
            for src, dest in zip(online_policy_variables, target_policy_variables):
                dest.assign(src)
        # Make online critic -> target critic network update ops.
        if tf.math.mod(self._num_steps, self._target_critic_update_period) == 0:
            for src, dest in zip(online_critic_variables, target_critic_variables):
                dest.assign(src)

        self._num_steps.assign_add(1)

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.

        # multiple replay buffers implementation here.
        inputs = next(iterator)
        transitions: types.Transition = inputs.data

        # Get batch size and scalar dtype.
        batch_size = transitions.reward.shape[0]

        # Cast the additional discount to match the environment discount dtype.
        discount = tf.cast(self._discount, dtype=transitions.discount.dtype)

        with tf.GradientTape(persistent=True) as tape:
            # Maybe transform the observation before feeding into policy and critic.
            # Transforming the observations this way at the start of the learning
            # step effectively means that the policy and critic share observation
            # network weights.
            o_tm1 = self._observation_network(transitions.observation)
            # Scott: This is the raw observation from the replay server. # TODO for Kickstarting, use this observation instead!

            # This stop_gradient prevents gradients to propagate into the target
            # observation network. In addition, since the online policy network is
            # evaluated at o_t, this also means the policy loss does not influence
            # the observation network training.
            o_t = tf.stop_gradient(self._target_observation_network(transitions.next_observation))

            # Get online and target action distributions from policy networks
            target_action_distribution = self._target_policy_network(o_t)

            # We don't want to unpack a tuple when the network just returns a distribution

            if self._KL_regularized:
                online_result = self._policy_network(o_t, return_intentions_dist=True)
                if isinstance(online_result, tuple):
                    if len(online_result) == 2:
                        online_action_distribution, intentions_dist = online_result
                else:
                    # If online_result is not a tuple but we need intentions_dist for KL regularization
                    # We have a problem - let's raise a more helpful error
                    raise ValueError(
                        f"Expected policy network to return a tuple with intentions_dist when KL_regularized=True, "
                        f"but got {type(online_result).__name__}"
                    )
            else:
                online_action_distribution = self._policy_network(o_t)
                # Handle case where policy returns a tuple even when we don't request intentions
                if isinstance(online_action_distribution, tuple):
                    online_action_distribution = online_action_distribution[0]

            # Sample actions to evaluate policy; of size [N, B, ...].
            sampled_actions = target_action_distribution.sample(self._num_samples)

            # Tile embedded observations to feed into the target critic network.
            # Note: this is more efficient than tiling before the embedding layer.
            tiled_o_t = tf2_utils.tile_tensor(o_t, self._num_samples)  # [N, B, ...]

            # Compute target-estimated distributional value of sampled actions at o_t.
            sampled_q_t_distributions = self._target_critic_network(
                # Merge batch dimensions; to shape [N*B, ...].
                snt.merge_leading_dims(tiled_o_t, num_dims=2),
                snt.merge_leading_dims(sampled_actions, num_dims=2),
            )

            # Compute average logits by first reshaping them and normalizing them
            # across atoms.
            new_shape = [self._num_samples, batch_size, -1]  # [N, B, A]
            sampled_logits = tf.reshape(sampled_q_t_distributions.logits, new_shape)
            sampled_logprobs = tf.math.log_softmax(sampled_logits, axis=-1)
            averaged_logits = tf.reduce_logsumexp(sampled_logprobs, axis=0)

            # Construct the expected distributional value for bootstrapping.
            q_t_distribution = networks.DiscreteValuedDistribution(
                values=sampled_q_t_distributions.values, logits=averaged_logits
            )

            # Compute online critic value distribution of a_tm1 in state o_tm1.
            q_tm1_distribution = self._critic_network(o_tm1, transitions.action)

            # Compute critic distributional loss.
            critic_loss = losses.categorical(
                q_tm1_distribution,
                transitions.reward,
                discount * transitions.discount,
                q_t_distribution,
            )
            critic_loss = tf.reduce_mean(critic_loss)

            # Compute Q-values of sampled actions and reshape to [N, B].
            sampled_q_values = sampled_q_t_distributions.mean()
            sampled_q_values = tf.reshape(sampled_q_values, (self._num_samples, -1))

            # Compute MPO policy loss.
            policy_loss, policy_stats = self._policy_loss_module(
                online_action_distribution=online_action_distribution,
                target_action_distribution=target_action_distribution,
                actions=sampled_actions,
                q_values=sampled_q_values,
            )

            # compute the kickstarting loss
            # Note: Have to do it here because the custom policy loss module is serialized in the head node via `DMPOConfig` and
            # we cannot serialize the checkpoint object
            loss_policy_teacher = 0
            # Compute the expert distillation loss
            if self._kickstart_teacher_policy is not None:
                teacher_distribution = self._kickstart_teacher_policy(o_tm1)
                kl_teacher_student = teacher_distribution.distribution.kl_divergence(
                    online_action_distribution.distribution
                )
                loss_policy_teacher = kl_teacher_student * self._kickstart_epsilon
            policy_loss += loss_policy_teacher
            policy_stats["loss_kickstart"] = tf.reduce_mean(loss_policy_teacher)
            print("DEBUG: IN LEARNER KL Weights: ", self._KL_weights)
            # compute the KL regularization costs
            if self._KL_regularized:
                KL_intention = intentions_dist.kl_divergence(self._intention_std_normal_dist)
                KL_action = online_action_distribution.kl_divergence(self._action_std_normkal_dist)
                # apply beta weights to the KL loss
                KL_intention_loss = tf.reduce_mean(KL_intention * self._KL_weights[0])
                KL_action_loss = tf.reduce_mean(KL_action * self._KL_weights[1])
                print("DEBUG: KL INTENTION LOSS: ", KL_intention_loss)
                policy_stats["intention_KL_loss"] = KL_intention_loss
                policy_stats["action_KL_loss"] = KL_action_loss
                policy_loss += KL_intention_loss + KL_action_loss

        # For clarity, explicitly define which variables are trained by which loss.
        critic_trainable_variables = (
            # In this agent, the critic loss trains the observation network.
            self._observation_network.trainable_variables
            + self._critic_network.trainable_variables
        )
        policy_trainable_variables = self._policy_network.trainable_variables
        # The following are the MPO dual variables, stored in the loss module.
        dual_trainable_variables = self._policy_loss_module.trainable_variables

        # Compute gradients.
        critic_gradients = tape.gradient(critic_loss, critic_trainable_variables)
        policy_gradients, dual_gradients = tape.gradient(
            policy_loss, (policy_trainable_variables, dual_trainable_variables)
        )

        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Maybe clip gradients.
        if self._clipping:
            policy_gradients = tuple(tf.clip_by_global_norm(policy_gradients, 40.0)[0])
            critic_gradients = tuple(tf.clip_by_global_norm(critic_gradients, 40.0)[0])

        # Apply gradients.
        self._critic_optimizer.apply(critic_gradients, critic_trainable_variables)
        self._policy_optimizer.apply(policy_gradients, policy_trainable_variables)
        self._dual_optimizer.apply(dual_gradients, dual_trainable_variables)

        # Losses to track.
        fetches = {
            "critic_loss": critic_loss,
            "policy_loss": policy_loss,
        }
        fetches.update(policy_stats)  # Log MPO stats.

        return fetches

    def step(self):
        # Run the learning step.
        for iterator in self._iterators:
            fetches = self._step(iterator)

            # Compute elapsed time.
            timestamp = time.time()
            elapsed_time = timestamp - self._timestamp if self._timestamp else 0
            self._timestamp = timestamp

            # Update our counts and record it.
            counts = self._counter.increment(steps=1, walltime=elapsed_time)
            fetches.update(counts)

            # Checkpoint and attempt to write the logs.
            if self._checkpointer is not None:
                self._checkpointer.save()

            if self._snapshotter is not None:
                if self._snapshotter.save():
                    # Increment the snapshot counter (directly in the snapshotter's path).
                    for path in list(self._snapshotter._snapshots.keys()):
                        snapshot = self._snapshotter._snapshots[path]  # noqa: F841
                        # Assume that path ends with, e.g., "/policy-17".
                        # Find sequence of digits at end of string.
                        current_counter = re.findall("[0-9]+$", path)[0]
                        new_path = path.replace(
                            "policy-only-no-obs-network-" + current_counter,
                            "policy-only-no-obs-network-" + str(int(current_counter) + 1),
                        )
                        new_path = new_path.replace(
                            "policy-" + current_counter,
                            "policy-" + str(int(current_counter) + 1),
                        )
                        if isinstance(self._target_policy_network, IntentionNetwork):
                            new_path = new_path.replace(
                                "policy-decoder-" + current_counter,
                                "policy-decoder-" + str(int(current_counter) + 1),
                            )
                            new_path = new_path.replace(
                                "policy-encoder-" + current_counter,
                                "policy-encoder-" + str(int(current_counter) + 1),
                            )
                        # Redirect snapshot to new key and delete old key.
                        self._snapshotter._snapshots[new_path] = self._snapshotter._snapshots.pop(path)
            try:
                fetches["actor_sps"] = fetches["actor_steps"] / (
                    fetches["learner_walltime"] + 1
                )  # calculate and report the sps of the actor
                fetches["learner_sps"] = fetches["learner_steps"] / (
                    fetches["learner_walltime"] + 1
                )  # calculate and report the sps fo the learner
            except KeyError:
                # sometime the first few fetches do not have some key.
                pass
            self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        self._counter.increment(get_variables_calls=1)
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]
