import tensorflow as tf
from typing import List, Dict, Any, Tuple, Union, Optional


def get_rodent_egocentric_obs_key() -> List[str]:
    """
    Returns the egocentric observation keys for the rodent.
    """
    return [
        "rodent/egocentric_camera",
    ]


def get_mouse_egocentric_obs_key() -> List[str]:
    """
    Returns the egocentric observation keys for the mouse.
    """
    return [
        "mouse/joint_angles",
        "mouse/joint_velocities",
    ]


def separate_observation(inputs):
    """
    Takes the input dictionary, separate it into task observations and egocentric observations.
    It concatenates the task and egocentric tensors independently and then concatenates these.

    Args:
        inputs: A dictionary of name to tensors, or a tensor.

    Returns:
        Tensor corresponding to concatenated task observations and egocentric observations.
    """
    # If input is not a dictionary, return it as is
    if not isinstance(inputs, dict):
        return inputs

    # Add batch dimension to all tensors if they don't have one
    inputs_with_batch = {}
    for k, v in inputs.items():
        # Make sure we're dealing with a tensor, not a tuple or other structure
        if isinstance(v, tuple) or isinstance(v, list):
            # Convert tuple/list to tensor if needed
            v = tf.convert_to_tensor(v)

        # Check tensor shape and add batch dimension if needed
        if hasattr(v, "shape") and hasattr(v.shape, "ndims"):
            if v.shape.ndims == 1:
                inputs_with_batch[k] = tf.expand_dims(v, 0)
            else:
                inputs_with_batch[k] = v
        else:
            # If it's not a tensor with shape attribute, convert it
            tensor_v = tf.convert_to_tensor(v)
            if tensor_v.shape.ndims == 1:
                inputs_with_batch[k] = tf.expand_dims(tensor_v, 0)
            else:
                inputs_with_batch[k] = tensor_v

    # Get keys for different observation types
    task_obs_keys = []
    egocentric_keys = get_mouse_egocentric_obs_key()

    # Filter which keys go into which category
    for k in inputs_with_batch.keys():
        if k not in egocentric_keys:
            task_obs_keys.append(k)

    # Collect tensors
    task_obs_tensors = [inputs_with_batch[k] for k in task_obs_keys if k in inputs_with_batch]
    egocentric_tensors = [inputs_with_batch[k] for k in egocentric_keys if k in inputs_with_batch]

    # Make sure we have compatible batch dimensions
    if task_obs_tensors and egocentric_tensors:
        task_batch_size = task_obs_tensors[0].shape[0]
        ego_batch_size = egocentric_tensors[0].shape[0]

        # If batch sizes don't match, adjust all tensors to batch size 1
        if task_batch_size != ego_batch_size:
            # Reshape all tensors to have batch size 1
            for i in range(len(task_obs_tensors)):
                shape = task_obs_tensors[i].shape.as_list()
                if shape[0] != 1:
                    task_obs_tensors[i] = tf.reshape(task_obs_tensors[i], [1, -1])

            for i in range(len(egocentric_tensors)):
                shape = egocentric_tensors[i].shape.as_list()
                if shape[0] != 1:
                    egocentric_tensors[i] = tf.reshape(egocentric_tensors[i], [1, -1])

    # Now concatenate each group separately
    task_obs_tensor = tf.concat(task_obs_tensors, axis=-1) if task_obs_tensors else tf.zeros((1, 0))
    egocentric_tensor = tf.concat(egocentric_tensors, axis=-1) if egocentric_tensors else tf.zeros((1, 0))

    # Finally concatenate both groups
    return tf.concat([task_obs_tensor, egocentric_tensor], axis=-1)
