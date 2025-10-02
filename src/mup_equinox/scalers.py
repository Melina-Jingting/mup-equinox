import equinox as eqx
import jax.random as jr
from .metadata import ParameterizationMetadata
from typing import Callable
import optax
from .utils import flexible_path_metadata_tree_map
import optax
import equinox as eqx
from .metadata import ParameterizationMetadata


def build_base_and_scaled_models(
    model_constructor: Callable[..., eqx.Module],
    base_kwargs: dict,
    scaled_kwargs: dict,
    rng_seed: int = 0,
) -> tuple[eqx.Module, eqx.Module, eqx.nn.State]:
    """
    Construct a base model and a width-scaled model (with state) from config.
    Args:
        model_constructor: Callable that builds an Equinox module, e.g. eqx.nn.MLP.
        base_cfg: Configuration object with 'args' dict and 'width_args' list.
        width_multiplier: Factor to multiply all width-related args listed in cfg.width_args.
    Returns:
        (base_model, scaled_model, state)
    """
    key = jr.PRNGKey(
        rng_seed
    )  # Same rng key for both models safe - only scaled model is used.
    base_model = model_constructor(**base_kwargs, key=key)
    scaled_model, state = eqx.nn.make_with_state(model_constructor)(
        **scaled_kwargs, key=key
    )
    return base_model, scaled_model, state


def scale_initializations(
    model: eqx.Module,
    metadata: eqx.Module,
    param_type: str = "muP_3",
    init_output_zero: bool = False,
):
    """
    Rescale a model’s parameter tree in-place according to MuP metadata.
    Parameters:
        model: Equinox module to adjust.
        metadata: pytree of ParameterizationMetadata sharing the model’s structure.
        param_type: 'muP_3' (default) or 'standard', controlling the scaling rules.
    Returns: new Equinox module combining scaled parameters with original static fields.
    Raises: ValueError for unsupported param_type. Leaves without metadata are returned untouched, so you can mix MuP-managed and fixed parameters.
    """

    params, static = eqx.partition(model, eqx.is_inexact_array)
    meta_params, _ = eqx.partition(
        metadata, lambda x: isinstance(x, ParameterizationMetadata)
    )

    def _init_leaf(param, meta):
        if meta is None:
            return param
        if param_type == "muP_3":
            if meta.is_output_weight:
                return (
                    param / (meta.width**0.5)
                    if init_output_zero == False
                    else param * 0.0
                )
            else:
                return param
        if param_type == "standard":
            return param
        raise ValueError(f"Unsupported param_type '{param_type}'")

    scaled_params = flexible_path_metadata_tree_map(
        _init_leaf,
        params,
        meta_params,
        is_leaf=lambda x: eqx.is_inexact_array(x)
        or isinstance(x, ParameterizationMetadata),
    )
    return eqx.combine(scaled_params, static)


def scale_gradients(
    metadata: eqx.Module,
    optimizer_type: str = "adam_like",
    param_type: str = "muP_3",
) -> optax.GradientTransformation:
    """
    Scale gradients according to Parameterization metadata.
    Args:
        metadata: An Equinox pytree of ParameterizationMetadata corresponding to the model parameters.
        optimizer: The name of the base optimizer to use (e.g., 'adam_like', 'sgd_like').
        param_type: The type of parameterization used ('muP_3' or 'standard').

    Returns:
        An Optax GradientTransformation that scales gradients according to the metadata.
    """

    def _scale_grad(grad, meta):
        if meta is None:
            return grad
        if param_type == "muP_3":
            if optimizer_type == "adam_like":
                return (
                    grad / meta.width
                    if (meta.is_output_weight or meta.is_hidden_weight)
                    else grad
                )
            if optimizer_type == "sgd_like":
                grad = (
                    grad * meta.width
                    if (meta.is_vector_like and not meta.is_output_weight)
                    else grad
                )
                grad = grad / meta.width if meta.is_output_weight else grad
                return grad
        elif param_type == "standard":
            return grad
        raise ValueError(
            f"""Unsupported param_type '{param_type}'. Only 'muP_3' and 'standard' are supported."""
        )

    def init_fn(params: optax.Params) -> optax.OptState:
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        del params

        scaled_updates = flexible_path_metadata_tree_map(
            _scale_grad,
            updates,
            metadata,
            is_leaf=lambda x: eqx.is_inexact_array(x)
            or isinstance(x, ParameterizationMetadata),
        )
        return scaled_updates, state

    return optax.GradientTransformation(init_fn, update_fn)
