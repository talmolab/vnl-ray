import tensorflow as tf
import logging


def load_checkpoint_with_expect_partial(checkpoint, checkpoint_to_load, print_fn=print):
    """
    Load a checkpoint with expect_partial() to suppress warnings about optimizer states.

    Args:
        checkpoint: TensorFlow checkpoint object
        checkpoint_to_load: Path to checkpoint file
        print_fn: Function to use for printing status

    Returns:
        Status object from checkpoint loading
    """
    if checkpoint_to_load is None:
        return None

    try:
        status = checkpoint.restore(checkpoint_to_load).expect_partial()
        print_fn(f"Loaded checkpoint from: {checkpoint_to_load}")
        return status
    except Exception as e:
        logging.error(f"Error loading checkpoint from {checkpoint_to_load}: {e}")
        return None
