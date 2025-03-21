from acme.tf import utils as tf2_utils
from acme.tf import networks
import tensorflow as tf
import sonnet as snt
from vnl_ray.agents.policy_network_activations import MLP_activations
from typing import List, Sequence, Callable, Optional, Dict
import tensorflow_probability as tfp
from agents.utils_sonnet import ReLU, NormalTanhDistribution, Sequential
import h5py
import functools


def _uniform_initializer():
    return tf.initializers.VarianceScaling(distribution="uniform", mode="fan_out", scale=0.333)


def recursively_print_h5(group, indent=0):
    for key in group:
        item = group[key]
        prefix = " " * indent + key
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}: shape={item.shape}, dtype={item.dtype}")
            # Uncomment next line to print the dataset's value:
            # print(item[()])
        elif isinstance(item, h5py.Group):
            print(f"{prefix} (Group)")
            recursively_print_h5(item, indent + 4)


def load_decoder_weights(h5_path):
    weights = {}
    with h5py.File(h5_path, "r") as hf:
        decoder = hf["decoder"]
        for key in decoder.keys():
            group = decoder[key]
            if "bias" in group and "kernel" in group:
                weights[key] = {
                    "bias": group["bias"][:],
                    "kernel": group["kernel"][:],
                }
    return weights


def load_params(path):
    """Takes in an h5 path and returns a dictionary of params"""
    import h5py

    params = {}
    with h5py.File(path, "r") as hf:
        # Optional: print structure for debugging
        # recursively_print_h5(hf, indent=0)
        # print(f"decoder keys: {list(hf['decoder'].keys())}")

        # Load the full hierarchy into a dictionary
        def extract_group(group, prefix=""):
            result = {}
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    result[key] = item[()]
                elif isinstance(item, h5py.Group):
                    result[key] = extract_group(item)
            return result

        params = extract_group(hf)

    return params


class EncoderJAX(Sequential):
    """
    Encoder Object Loaded from the JAX checkpoint, produced by MJX
    """

    def __init__(
        self,
        layer_sizes: List[int],
        layer_norm: bool,
        intention_size: int,
        min_scale: float,
        module_name: str = "EncoderJAX",
    ):
        # need to separate the fc layer and the mean/var layer
        self.num_layers = len(layer_sizes)
        self.layers: List[callable] = []  # will be a list of network_obj
        for i, size in enumerate(layer_sizes):
            name = f"hidden_{i}"
            linear = snt.Linear(size, name=name)
            self.layers.append(linear)
            if layer_norm:
                ln_name = f"LayerNorm_{i}"
                # if we want to freeze the decoder layernorm, use the checkpoint stats
                # in __call__ of the layer norm.
                ln = snt.LayerNorm(axis=slice(1, None), create_scale=False, create_offset=False, name=ln_name)
                self.layers.append(ln)
            self.layers.append(ReLU(name=f"relu_{i}"))

        # linear layer for the stochastic mean and std
        fc2_mean = snt.Linear(intention_size, name=f"fc2_mean")
        fc2_logvar = snt.Linear(intention_size, name=f"fc2_logvar")
        self.layers.append(fc2_mean)
        self.layers.append(fc2_logvar)
        stochastic_actions = NormalTanhDistribution(intention_size, min_scale)
        self.layers.append(stochastic_actions)
        self.name_to_layer = {l.name: l for l in self.layers}
        super().__init__(layers=self.layers, name=module_name, need_activation=True)

    def __call__(self, inputs, return_activations=False, *args, **kwargs):
        """
        Call method of the sonnet module. If a layer is a tuple, it will return two outputs correspondingly. Only the last layer is allowed with a tuple
        """
        activations = {}
        outputs = inputs
        for i in range(self.num_layers):
            mod = self.name_to_layer[f"hidden_{i}"]
            outputs = mod(outputs)
            activations[mod.name] = outputs
            mod = self.name_to_layer[f"relu_{i}"]  # hard coded relu for now
            outputs = mod(outputs)
            activations[mod.name] = outputs  # record the activations
        mean = self.name_to_layer["fc2_mean"](outputs)
        activations["fc2_mean"] = mean
        log_var = self.name_to_layer["fc2_logvar"](outputs)
        activations["fc2_logvar"] = log_var
        input_to_dist = tf.concat([mean, log_var], axis=-1)
        outputs = self.name_to_layer["NormalTanhDist"](input_to_dist)
        activations["NormalTanhDist"] = outputs
        if return_activations:
            return outputs, activations
        return outputs

    def load_from_h5_checkpoint(self, path):
        params = load_params(path)
        network = params[1]["params"]
        encoder_cpts = network["encoder"]
        transfer_names = list(filter(lambda x: "relu" not in x and "NormalTanhDist" not in x, self._layer_names))

        if set(transfer_names) != set(encoder_cpts.keys()):
            raise ValueError(
                f"Key mismatch between target model: {set(transfer_names)} and the checkpoint: {set(encoder_cpts.keys())}"
            )

        for key in transfer_names:
            module = self.name_to_layer[key]
            if "hidden" in key:
                module.w.assign(encoder_cpts[key]["kernel"])
                module.b.assign(encoder_cpts[key]["bias"])
            if "LayerNorm" in key:
                idx = self._layer_names.index(key)
                # change the layers in place.
                par = functools.partial(module, scale=encoder_cpts[key]["scale"], offset=encoder_cpts[key]["bias"])
                self.layers[idx] = par
                self.name_to_layer[key] = par
            print(f"Layer: {key} transferred!")


class DecoderJAX(Sequential):
    """
    Decoder object loaded from the JAX checkpoint, produced by MuJoCo MJX
    """

    def __init__(
        self,
        layer_sizes: List[int],
        layer_norm: bool,
        action_size: int,
        min_scale: float,
        module_name: str = "DecoderJAX",
    ):
        self.layers: List[callable] = []  # will be a list of network_obj
        for i, size in enumerate(layer_sizes):
            name = f"hidden_{i}"
            linear = snt.Linear(size, name=name)
            self.layers.append(linear)
            if layer_norm:
                ln_name = f"LayerNorm_{i}"
                # if we want to freeze the decoder layernorm, use the checkpoint stats
                # in __call__ of the layer norm.
                ln = snt.LayerNorm(axis=slice(1, None), create_scale=False, create_offset=False, name=ln_name)
                self.layers.append(ln)
            self.layers.append(ReLU(name=f"relu_{i}"))
        # linear layer for the stochastic mean and std
        linear = snt.Linear(action_size * 2, name=f"hidden_{len(layer_sizes)}")
        self.layers.append(linear)
        stochastic_actions = NormalTanhDistribution(action_size, min_scale)
        self.layers.append(stochastic_actions)
        self.name_to_layer = {l.name: l for l in self.layers}
        super().__init__(layers=self.layers, name=module_name, need_activation=True)

    def __call__(self, inputs, return_activations=False, *args, **kwargs):
        """Call method of the sonnet module."""
        return super().__call__(inputs, return_activations=return_activations)

    def load_from_h5_checkpoint(self, path):
        """Load model weights from h5 checkpoint file."""
        print(f"Loading decoder weights from checkpoint: {path}")
        params = load_params(path)

        # Get the decoder part of the parameters
        decoder_cpts = None

        # Try different possible structures in the h5 file
        if "decoder" in params:
            decoder_cpts = params["decoder"]
            print(f"Found decoder weights directly in top level 'decoder' key")
        elif 1 in params and "params" in params[1] and "decoder" in params[1]["params"]:
            # Structure might be nested like in JAX checkpoints
            decoder_cpts = params[1]["params"]["decoder"]
            print(f"Found decoder weights in nested JAX-style checkpoint structure")

        if decoder_cpts is None:
            print("Checkpoint structure:")
            for k in params:
                print(f"  {k}: {type(params[k])}")
            raise ValueError(f"Could not find decoder weights in checkpoint: {path}")

        # Get the module names we need to transfer (excluding activations and distribution layers)
        transfer_names = list(filter(lambda x: "relu" not in x and "NormalTanhDist" not in x, self._layer_names))
        print(f"Transferable layer names: {transfer_names}")
        print(f"Checkpoint layer names: {list(decoder_cpts.keys())}")

        if set(transfer_names) != set(decoder_cpts.keys()):
            raise ValueError(
                f"Key mismatch between target model: {set(transfer_names)} and the checkpoint: {set(decoder_cpts.keys())}"
            )

        print(f"Transferring weights for {len(transfer_names)} layers...")
        for key in transfer_names:
            module = self.name_to_layer[key]
            if "hidden" in key:
                module.w.assign(decoder_cpts[key]["kernel"])
                module.b.assign(decoder_cpts[key]["bias"])
                print(f"✓ Transferred weights for linear layer: {key}")
            if "LayerNorm" in key:
                idx = self._layer_names.index(key)
                # change the layers in place.
                par = functools.partial(module, scale=decoder_cpts[key]["scale"], offset=decoder_cpts[key]["bias"])
                self.layers[idx] = par
                self.name_to_layer[key] = par
                print(f"✓ Transferred weights for normalization layer: {key}")

        print(f"✓ Successfully loaded all decoder weights from: {path}")


class IntentionNetworkJAX(snt.Module):
    """encoder decoder now have the same size from the policy layer argument, decoder + latent"""

    def __init__(
        self,
        action_size: int,
        intention_size: int,
        task_obs_size: int,
        min_scale: float,
        encoder_layer_sizes: List[int],
        decoder_layer_sizes: List[int],
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
        self.encoder = EncoderJAX(
            encoder_layer_sizes, layer_norm=True, intention_size=intention_size, min_scale=min_scale
        )
        self.decoder = DecoderJAX(decoder_layer_sizes, layer_norm=True, action_size=action_size, min_scale=min_scale)

    def load_from_h5_checkpoint(self, path):
        # initialize the network with appropriate size
        self.encoder(tf.ones((1, self.task_obs_size), dtype=tf.float32))
        self.decoder(tf.ones((1, self.intention_size + 147), dtype=tf.float32))  # hardcoded!
        # load the checkpoint
        self.encoder.load_from_h5_checkpoint(path)
        self.decoder.load_from_h5_checkpoint(path)

    def __call__(self, observations, return_intentions_dist=False):
        """
        split the observation tensor to task obs and egocentric obs, and pass through
        the encoder -> intention -> decoder
        """
        # split the observation
        task_obs = observations[..., : self.task_obs_size]
        egocentric_obs = observations[..., self.task_obs_size :]
        # feed into the encoder
        # maybe this batch-concat can be taken off as it already in the encoder?
        intentions_dist = self.encoder(tf2_utils.batch_concat(task_obs))
        intentions = intentions_dist.sample()
        concatenated = tf.concat([intentions, egocentric_obs], axis=-1)
        actions = self.decoder(tf2_utils.batch_concat(concatenated))
        if return_intentions_dist:
            return actions, intentions_dist
        return actions


class LayerNormMLP_activations(snt.Module):
    """Feedforward MLP torso with initial layer-norm returning intermediate activations."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        w_init: Optional[snt.initializers.Initializer] = None,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.elu,
        activate_final: bool = False,
    ):
        super().__init__(name="layernorm_mlp_torso_activations")
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
            # Try to safely check if the layer's __call__ supports return_activations.
            try:
                supports_return = "return_activations" in layer.__call__.__code__.co_varnames
            except AttributeError:
                supports_return = False
            if supports_return:
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
        self.return_activations = False  # Changed from parameter value to always False
        self.use_multi_encoder = high_level_intention_size is not None
        self.mid_layer_sizes = mid_layer_sizes
        self.high_level_intention_size = high_level_intention_size
        if mid_layer_sizes is not None:
            self.high_level_encoder = TransparentSequential(
                [
                    tf2_utils.batch_concat,
                    MLP_activations(
                        output_sizes=encoder_layer_sizes,
                        w_init=None,
                        b_init=None,
                        with_bias=True,
                        activation=tf.nn.elu,
                        dropout_rate=None,
                        activate_final=True,
                    ),
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
                    MLP_activations(
                        output_sizes=mid_layer_sizes,
                        w_init=None,
                        b_init=None,
                        with_bias=True,
                        activation=tf.nn.elu,
                        dropout_rate=None,
                        activate_final=True,
                    ),
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
                    MLP_activations(
                        output_sizes=encoder_layer_sizes,
                        w_init=None,
                        b_init=None,
                        with_bias=True,
                        activation=tf.nn.elu,
                        dropout_rate=None,
                        activate_final=True,
                    ),
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

    def __call__(self, observations, return_intentions_dist=False, return_activations=False):
        """
        split the observation tensor to task obs and egocentric obs, and pass through
        the encoder -> intention -> decoder

        Args:
            observations: Input observations tensor
            return_intentions_dist: Whether to return the intention distribution
            return_activations: Whether to return activations (only for debugging/analysis)
        """
        task_obs = observations[..., : self.task_obs_size]
        egocentric_obs = observations[..., self.task_obs_size :]
        activations_dict = {}

        # Use explicit parameter value, not the instance attribute
        should_return_activations = return_activations or return_intentions_dist

        if self.use_multi_encoder:
            if should_return_activations:
                high_out, high_acts = self.high_level_encoder(task_obs, return_activations=True)
                activations_dict["high_level_encoder"] = high_acts
            else:
                high_out = self.high_level_encoder(task_obs)

            if should_return_activations:
                mid_out, mid_acts = self.mid_level_encoder(high_out, return_activations=True)
                activations_dict["mid_level_encoder"] = mid_acts
            else:
                mid_out = self.mid_level_encoder(high_out)

            intentions = mid_out
        else:
            if should_return_activations:
                intentions, enc_acts = self.encoder(task_obs, return_activations=True)
                activations_dict["encoder"] = enc_acts
            else:
                intentions = self.encoder(task_obs)

        # Store the intentions distribution only if explicitly requested
        if should_return_activations:
            activations_dict["intentions_dist"] = intentions

        if isinstance(intentions, tfp.distributions.Distribution):
            intentions_tensor = intentions.mean()
        else:
            intentions_tensor = intentions

        concatenated = tf.concat([intentions_tensor, egocentric_obs], axis=-1)

        if should_return_activations:
            actions, dec_acts = self.decoder(tf2_utils.batch_concat(concatenated), return_activations=True)
            activations_dict["decoder"] = dec_acts

            # Only return additional info if explicitly requested
            if return_intentions_dist:
                return actions, intentions_dist
            elif return_activations:
                return actions, activations_dict
        else:
            # Simple path for inference - just return the actions distribution
            actions = self.decoder(tf2_utils.batch_concat(concatenated))

        return actions
