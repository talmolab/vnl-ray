"""Utilities for swapping decoders in intention networks."""

import tensorflow as tf
from vnl_ray.agents.intention_network_base import load_params, DecoderJAX


def swap_decoder_with_jax(
    intention_network, decoder_h5_path, action_size, layer_sizes, min_scale=0.1, freeze_weights=True
):
    """
    Swaps the decoder in an IntentionNetwork with a JAX decoder loaded from h5.

    Args:
        intention_network: The intention network instance to modify
        decoder_h5_path: Path to the h5 file containing decoder weights
        action_size: Size of action space
        layer_sizes: List of layer sizes for the decoder
        min_scale: Minimum scale for the decoder's NormalTanhDistribution
        freeze_weights: Whether to mark the decoder weights as non-trainable

    Returns:
        The modified intention network with JAX decoder
    """
    # Create a JAX decoder with the right dimensions
    jax_decoder = DecoderJAX(layer_sizes=layer_sizes, layer_norm=True, action_size=action_size, min_scale=min_scale)

    # Initialize the decoder with some dummy input of the correct size.
    # (intention size + egocentric obs size)
    input_size = intention_network.intention_size + (
        intention_network._observation_network.output_size
        if hasattr(intention_network._observation_network, "output_size")
        else 147
    )
    jax_decoder(tf.ones((1, input_size), dtype=tf.float32))

    # Load the weights from the h5 checkpoint
    jax_decoder.load_from_h5_checkpoint(decoder_h5_path)

    # Replace the decoder in the intention network
    original_decoder_type = type(intention_network.decoder).__name__
    intention_network.decoder = jax_decoder

    if freeze_weights:
        # Freeze each trainable variable and mark the entire module as non-trainable.
        for var in jax_decoder.trainable_variables:
            var._trainable = False
        jax_decoder.trainable = False

    return intention_network
