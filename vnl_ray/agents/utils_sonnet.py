import sonnet as snt
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from functools import partial as p

tfd = tfp.distributions


class MLPActivations(snt.nets.MLP):
    """A multi-layer perceptron module with layer activations tracking."""

    def __init__(
        self,
        output_sizes,
        w_init=None,
        b_init=None,
        with_bias=True,
        activation=tf.nn.relu,
        dropout_rate=None,
        activate_final=False,
        name=None,
    ):
        """
        Init function for this custom MLP Activation
        """
        super().__init__(output_sizes, w_init, b_init, with_bias, activation, dropout_rate, activate_final, name)

    def __call__(self, inputs: tf.Tensor, is_training=None, return_activations=False) -> tf.Tensor:
        """Connects the module to some inputs.

        Args:
          inputs: A Tensor of shape `[batch_size, input_size]`.
          is_training: A bool indicating if we are currently training. Defaults to
            `None`. Required if using dropout.
          return_activations: If True, returns a dictionary of layer activations.

        Returns:
          output: The output of the model of size `[batch_size, output_size]`.
          activations (optional): A dictionary of activations from each layer.
        """
        use_dropout = self._dropout_rate not in (None, 0)
        if use_dropout and is_training is None:
            raise ValueError("The `is_training` argument is required when dropout is used.")
        elif not use_dropout and is_training is not None:
            raise ValueError("The `is_training` argument should only be used with dropout.")

        num_layers = len(self._layers)
        activations = {}  # Dictionary to store layer activations

        for i, layer in enumerate(self._layers):
            inputs = layer(inputs)  # Apply the linear transformation
            if i < (num_layers - 1) or self._activate_final:
                # Only perform dropout if we are activating the output.
                if use_dropout and is_training:
                    inputs = tf.nn.dropout(inputs, rate=self._dropout_rate)
                inputs = self._activation(inputs)  # Apply the activation function

            if return_activations:
                # Store the output of each layer in the dictionary
                activations[f"layer_{i}"] = inputs

        if return_activations:
            return inputs, activations
        return inputs


class Sequential(snt.Sequential):
    """
    sub-class of `snt.Sequential` to make
    intermediate activation accessible
    """

    def __init__(self, layers, name=None, need_activation=False):
        """
        Initializer of the custom sequential methods

        layers: list of layers that the neural network is operating on
        name: name of the Sequential Module
        activation bool: whether it returns activations of each intermediate layers
        """
        super().__init__(layers, name)
        self._layer_names = [l.name for l in layers]
        if None in self._layer_names and need_activation:
            raise ValueError(
                f"User requested activations, did not give name to each layer. Current layer name is: {self._layer_names}"
            )

    def __call__(self, inputs, return_activations=False, *args, **kwargs):
        """
        Call method of the sonnet module. If a layer is a tuple, it will return two outputs correspondingly. Only the last layer is allowed with a tuple
        """
        activations = {}
        outputs = inputs
        for i, mod in enumerate(self._layers):
            if i == 0:
                # Pass additional arguments to the first layer.
                outputs = mod(outputs, *args, **kwargs)
                act = outputs  # last layer output is activation of this layer
            else:
                if isinstance(mod, Sequential) and return_activations:
                    # nested Sequential, pass the return_activations and get activations.
                    outputs, act = mod(
                        outputs, return_activations
                    )  # output is last layer, act is every activation includes last layer.
                else:
                    outputs = mod(outputs)
                    act = outputs
            activations[mod.name] = act  # record the activations
        if return_activations:
            return outputs, activations
        return outputs


class ReLU(snt.Module):
    def __init__(self, name=None):  # Add a name argument
        super().__init__(name=name)  # Pass the name to the superclass

    def __call__(self, x):
        return tf.nn.relu(x)


class NormalTanhDistribution(snt.Module):
    """ """

    def __init__(self, action_size, min_std, name="NormalTanhDist"):
        super().__init__(name)
        self._action_size = action_size
        self._min_std = min_std

    def __call__(self, x):
        if x.shape[-1] != self._action_size * 2:
            raise ValueError("Inconsistent input to distribution: input does not equals to double the action size")
        # Replace numpy split with tensorflow split
        mean, scale = tf.split(x, 2, axis=-1)
        scale = tf.nn.softplus(scale) + self._min_std
        return tfd.MultivariateNormalDiag(loc=mean, scale_diag=scale)
