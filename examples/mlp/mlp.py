
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
import jax
from jaxtyping import Array

class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(
        self, in_dim: int, out_dim: int, width: int, depth: int, *, key
    ):
        keys = jr.split(key, depth + 1)
        self.layers = [
            eqx.nn.Linear(
                in_dim if i == 0 else width,
                out_dim if i == depth - 1 else width,
                key=keys[i],
            )
            for i in range(depth)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))

        x = self.layers[-1](x)
        return x

    def get_activations(
        self, x: Array, *, layer_keys: list[str] | str | None = None, return_outputs=False, **kwargs
    ) -> dict[str, Array] | tuple[Array, dict[str, Array]]:
        
        activations = {}
        def _capture(k, v):
            if layer_keys is None or layer_keys == "all" or k in layer_keys:
                activations[k] = v
                return

        for i, layer in enumerate(self.layers[:-1]):
            x = jnp.tanh(layer(x))
            _capture(f"layers.{i}", x)

        x = self.layers[-1](x)
        _capture("logits", x)
        if return_outputs:
            return x, activations
        else:
            return activations