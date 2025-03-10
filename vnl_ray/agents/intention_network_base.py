from acme.tf import utils as tf2_utils
from acme.tf import networks
import tensorflow as tf
import sonnet as snt
from vnl_ray.agents.policy_network_activations import MLP_activations
from typing import List, Sequence, Callable, Optional, Dict

def _uniform_initializer():
    return tf.initializers.VarianceScaling(distribution="uniform", mode="fan_out", scale=0.333)

class LayerNormMLP_activations(snt.Module):
    """Feedforward MLP torso with initial layer-norm returning intermediate activations."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        w_init: Optional[snt.initializers.Initializer] = None,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.elu,
        activate_final: bool = False
    ):
        super().__init__(name='layernorm_mlp_torso_activations')
        w_init = w_init or _uniform_initializer()
        self._activate_final = activate_final
        # First layer: Linear + LayerNorm + tanh.
        self._linear = snt.Linear(layer_sizes[0], w_init=w_init)
        self._layer_norm = snt.LayerNorm(axis=slice(1, None), create_scale=True, create_offset=True)
        self._pre_activation = tf.nn.tanh
        # Following MLP layers.
        self._mlp_layers = []
        for i, size in enumerate(layer_sizes[1:]):
            self._mlp_layers.append(snt.Linear(size, w_init=w_init))
        self._activation = activation

    def __call__(self, observations: tf.Tensor, return_activations: bool = False) -> tf.Tensor:
        # Batch-concatenate observations.
        x = tf2_utils.batch_concat(observations)
        activations: Dict[str, tf.Tensor] = {}
        # First linear.
        x = self._linear(x)
        if return_activations:
            activations["linear"] = x
        # Layer norm.
        x = self._layer_norm(x)
        if return_activations:
            activations["layer_norm"] = x
        # tanh activation.
        x = self._pre_activation(x)
        if return_activations:
            activations["pre_activation_tanh"] = x
        # Process through MLP layers.
        for i, layer in enumerate(self._mlp_layers):
            x = layer(x)
            # Activate unless final layer and activate_final is False.
            if i < len(self._mlp_layers) - 1 or self._activate_final:
                x = self._activation(x)
            if return_activations:
                activations[f"mlp_layer_{i}"] = x
        if return_activations:
            return x, activations
        return x


class TransparentSequential(snt.Module):
    def __init__(self, layers):
        super().__init__()
        self._layers = layers

    def __call__(self, inputs, return_activations=False):
        activations = {}
        x = inputs
        for i, layer in enumerate(self._layers):
            if hasattr(layer, "__call__") and "return_activations" in layer.__call__.__code__.co_varnames:
                if return_activations:
                    x, act = layer(x, return_activations=True)
                    activations[f"layer_{i}"] = act
                else:
                    x = layer(x)
            else:
                x = layer(x)
                if return_activations:
                    activations[f"layer_{i}"] = x
        if return_activations:
            return x, activations
        return x


class Decoder(snt.Module):
    """
    separate decoder structure for skills reuses
    """

    def __init__(
        self,
        decoder_layer_sizes,
        action_size,
        min_scale,
        tanh_mean,
        init_scale,
        fixed_scale,
        use_tfd_independent,
    ):
        """
        decoder_layer_sizes: the size of the decoder layer
        action_size: the action output size
        min_scale: the minimal scale for the action space.

        returns a tfd.distribution
        """
        super().__init__()
        self.MLP = LayerNormMLP_activations(layer_sizes=decoder_layer_sizes, activate_final=True)
        self.stochastic_actions = networks.MultivariateNormalDiagHead(
            action_size,
            min_scale=min_scale,
            tanh_mean=tanh_mean,
            init_scale=init_scale,
            fixed_scale=fixed_scale,
            use_tfd_independent=use_tfd_independent,
        )

    def __call__(self, inputs, return_activations=False):
        """
        Tether the module together to be the decoder module. Can record the activation if
        `return_activations` is True
        """
        if return_activations:
            x, mlp_acts = self.MLP(inputs, return_activations=True)
        else:
            x = self.MLP(inputs)
        out = self.stochastic_actions(x)
        if return_activations:
            return out, {"mlp": mlp_acts}
        return out


class IntentionNetwork(snt.Module):
    """encoder decoder now have the same size from the policy layer argument, decoder + latent"""

    def __init__(
        self,
        action_size: int,
        intention_size: int,
        task_obs_size: int,
        min_scale: float,
        tanh_mean: bool,
        init_scale: float,
        action_dist_scale: float,
        use_tfd_independent: bool,
        encoder_layer_sizes: List[int],
        decoder_layer_sizes: List[int],
        mid_layer_sizes: List[int] = None,
        high_level_intention_size: int | None = None,
        return_activations: bool = False,
    ):
        """
        action_size: the action size for the output of the network
        intention_size: specify the size of the intention stochastic layer
        task_obs_size: the tasks specific observation size.
        min_scale: float: specify the minimal scale of the stochastic layer
        tanh_mean: bool, whether we apply tanh_mean layer
        init_scale: float, the scale of of the stochastic layer that we initialize to
        action_dist_scale: float, the scale of the action output layers
        use_tfd_independent: bool, whether we use tfd independent to model the stochastic layer
        encoder_layer_sizes: List[int], specifies the layer sizes of the encoder
        decoder_layer_sizes: List[int], specifies the layer sizes of the decoder
        mid_layer_sizes: List[int], if specified, will create an additional high level motor intention stochastic layer
            useful in skill transfer of the multi-tasks
        high_level_intention_size: int, specify the high level intention stochastic layer sizes.
        """

        super().__init__()
        self.task_obs_size = task_obs_size
        self.action_size = action_size
        self.intention_size = intention_size
        self.return_activations = return_activations
        self.use_multi_encoder = high_level_intention_size is not None
        self.mid_layer_sizes = mid_layer_sizes
        self.high_level_intention_size = high_level_intention_size
        if mid_layer_sizes is not None:
            self.high_level_encoder = TransparentSequential(
                [
                    tf2_utils.batch_concat,
                    MLP_activations(layer_sizes=encoder_layer_sizes, activation=tf.nn.elu, activate_final=True),
                    networks.MultivariateNormalDiagHead(
                        num_dimensions=high_level_intention_size,
                        min_scale=min_scale,
                        tanh_mean=tanh_mean,
                        init_scale=init_scale,
                        fixed_scale=False,
                        use_tfd_independent=use_tfd_independent,
                    ),
                ]
            )
            self.mid_level_encoder = TransparentSequential(
                [
                    MLP_activations(layer_sizes=mid_layer_sizes, activation=tf.nn.elu, activate_final=True),
                    networks.MultivariateNormalDiagHead(
                        intention_size,
                        min_scale=min_scale,
                        tanh_mean=tanh_mean,
                        init_scale=init_scale,
                        fixed_scale=False,
                        use_tfd_independent=use_tfd_independent,
                    ),
                ]
            )
        else:
            self.encoder = TransparentSequential(
                [
                    tf2_utils.batch_concat,
                    MLP_activations(layer_sizes=encoder_layer_sizes, activation=tf.nn.elu, activate_final=True),
                    networks.MultivariateNormalDiagHead(
                        intention_size,
                        min_scale=min_scale,
                        tanh_mean=tanh_mean,
                        init_scale=init_scale,
                        fixed_scale=False,
                        use_tfd_independent=use_tfd_independent,
                    ),
                ]
            )

        self.decoder = Decoder(
            decoder_layer_sizes=decoder_layer_sizes,
            action_size=action_size,
            min_scale=min_scale,
            tanh_mean=tanh_mean,
            init_scale=action_dist_scale,
            fixed_scale=True,
            use_tfd_independent=use_tfd_independent,
        )

    def __call__(self, observations, return_intentions_dist=False):
        """
        split the observation tensor to task obs and egocentric obs, and pass through 
        the encoder -> intention -> decoder
        """
        task_obs = observations[..., : self.task_obs_size] 
        egocentric_obs = observations[..., self.task_obs_size :]
        activations_dict = {}

        if self.use_multi_encoder:
            if self.return_activations or return_intentions_dist:
                high_out, high_acts = self.high_level_encoder(task_obs, return_activations=True)
                activations_dict["high_level_encoder"] = high_acts
            else:
                high_out = self.high_level_encoder(task_obs)
            if self.return_activations or return_intentions_dist:
                mid_out, mid_acts = self.mid_level_encoder(high_out, return_activations=True)
                activations_dict["mid_level_encoder"] = mid_acts
            else:
                mid_out = self.mid_level_encoder(high_out)
            intentions = mid_out
        else:
            if self.return_activations or return_intentions_dist:
                intentions, enc_acts = self.encoder(task_obs, return_activations=True)
                activations_dict["encoder"] = enc_acts
            else:
                intentions = self.encoder(task_obs)
        concatenated = tf.concat([intentions, egocentric_obs], axis=-1)
        if self.return_activations or return_intentions_dist:
            actions, dec_acts = self.decoder(tf2_utils.batch_concat(concatenated), return_activations=True)
            activations_dict["decoder"] = dec_acts
            return actions, activations_dict
        return self.decoder(tf2_utils.batch_concat(concatenated))
