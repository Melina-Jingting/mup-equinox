
import equinox as eqx
import jax.random as jr
import jax
from typing import Sequence

class CNN(eqx.Module):
    conv_layers: list[eqx.nn.Conv2d]
    linear_layer: eqx.nn.Linear

    def __init__(
        self, img_dim: Sequence[int], out_dim: int, width: int, depth: int, *, key
    ):
        keys = jr.split(key, depth + 1)
        self.conv_layers = [
            eqx.nn.Conv2d(
                in_channels = 1 if i == 0 else width,
                out_channels = width,
                kernel_size=(3, 3),
                stride= (2, 2),
                padding=1,
                key=keys[i],
            )
            for i in range(depth)
        ]
        self.linear_layer = eqx.nn.Linear(
            in_features = (width * self._conv_output_dim(img_dim[0], depth=depth) * self._conv_output_dim(img_dim[1], depth=depth)),
            out_features = out_dim,
            key=keys[-1],
        )
        
    def _conv_output_dim(self, in_dim, kernel=3, pad=1, stride=2, depth=1):
        x = in_dim
        for _ in range(depth):
            x = (x + 2*pad - kernel) // stride + 1
        return x

    def __call__(self, x):
        for layer in self.conv_layers:
            x = jax.nn.gelu(layer(x))
        x = x.reshape(-1)  # flatten
        x = self.linear_layer(x)
        return x
    
    def get_activations(
        self, x, *, layer_keys: list[str] | str | None = None, return_outputs=False, **kwargs
    ) -> dict[str, jax.Array] | tuple[jax.Array, dict[str, jax.Array]]:
        
        activations = {}
        def _capture(k, v):
            if layer_keys is None or layer_keys == "all" or k in layer_keys:
                activations[k] = v
                return

        for i, layer in enumerate(self.conv_layers):
            x = jax.nn.gelu(layer(x))
            _capture(f"conv_layers.{i}", x)

        x = x.reshape(-1)  # flatten
        x = self.linear_layer(x)
        _capture("logits", x)
        if return_outputs:
            return x, activations
        else:
            return activations
    