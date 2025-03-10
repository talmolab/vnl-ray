import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils
from typing import Sequence, Callable, Optional, Dict

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
