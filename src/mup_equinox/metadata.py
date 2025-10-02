import equinox as eqx
from jaxtyping import Array
from .utils import flexible_path_metadata_tree_map

class ParameterizationMetadata(eqx.Module):
    """Model Parameters with Dimension Metadata for Maximal Update Parameterization (MuP)."""

    value: Array  # parameter value
    dims: tuple[float | None, ...] = eqx.field(
        static=True
    )  # mup dims, None for finite dims

    @property
    def ndims(self) -> int:
        """Number of infinite dimensions."""
        return len(list(filter(None, self.dims)))

    @property
    def width(self) -> float:
        """Width of the final transformation."""
        assert self.ndims <= 2, "Only supports a maximum of two infinite dimensions"
        # The width is defined as the size of the last infinite dimension,
        # or 1.0 if there are no infinite dimensions.
        return next(filter(None, reversed(self.dims)), 1.0)

    @property
    def is_vector_like(self) -> bool:
        """A parameter is vector-like if it has one infinite dimension."""
        return self.ndims == 1

    @property
    def is_matrix_like(self) -> bool:
        """A parameter is matrix-like if it has two infinite dimensions."""
        return self.ndims == 2

    @property
    def is_scalar_like(self) -> bool:
        """A parameter is scalar-like if it has zero infinite dimensions."""
        return self.ndims == 0

    @property
    def is_input_weight(self) -> bool:
        """A weight that maps from a finite dimension to an infinite dimension."""
        return self.is_vector_like and self.dims[-1] is not None

    @property
    def is_output_weight(self) -> bool:
        """A weight that maps from an infinite dimension to a finite dimension."""
        return self.is_vector_like and self.dims[-1] is None

    @property
    def is_hidden_weight(self) -> bool:
        """A weight that maps from an infinite dimension to an infinite dimension."""
        return self.is_matrix_like


def build_param_metadata(
    base_model: eqx.Module, target_model: eqx.Module
) -> eqx.Module:
    """
    Generate ParameterizationMetadata leaves describing how a target model’s parameters scale relative to a base model.
    Parameters:
        base_model: reference Equinox module built at base width.
        target_model: width-scaled Equinox module with matching logical layout.
    Returns:
        pytree of ParameterizationMetadata aligned with the models’ parameter trees.
    Notes:
        assumes earlier validation guarantees structural parity; only relative dimension ratios are computed. Raises the underlying helper errors on mismatch.
    """
    base_params, _ = eqx.partition(base_model, eqx.is_array_like)
    target_params, _ = eqx.partition(target_model, eqx.is_array_like)

    def _get_metadata_leaf(base_param, target_param):
        dims = []

        # reversed because weight shapes are (out_features, in_features)
        for base_dim, target_dim in zip(
            reversed(base_param.shape), reversed(target_param.shape)
        ):
            dims.append(
                target_dim / base_dim
            ) if target_dim != base_dim else dims.append(None)
        return ParameterizationMetadata(value=target_param, dims=tuple(dims))

    meta_params = flexible_path_metadata_tree_map(
        _get_metadata_leaf,
        base_params,
        target_params,
        check_type=True,
        check_ndims=True,
    )
    return meta_params
