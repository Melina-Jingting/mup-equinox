from mup_equinox import (
    TrainingConfig,
    CoordinateCheckConfig,
    ModelFactory,
    OptimizerFactory,
    CoordinateCheckRunner,
)
import equinox as eqx
import optax
from typing import Iterator, TypedDict
from jaxtyping import Array, Float, Integer
import jax.random as jr
import jax.numpy as jnp
import jax


class TrainingBatch(TypedDict):
    inputs: Float[Array, "batch 4"]
    labels: Integer[Array, "batch 1"]


class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(
        self, in_size: int, out_size: int, width_size: int, depth: int, *, key
    ):
        keys = jr.split(key, depth + 1)
        self.layers = [
            eqx.nn.Linear(
                in_size if i == 0 else width_size,
                out_size if i == depth - 1 else width_size,
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


def make_dataset(batch_size: int = 32, rng_seed: int = 0) -> Iterator[TrainingBatch]:
    key = jr.PRNGKey(rng_seed)
    while True:
        key, input_subkey, label_subkey = jr.split(key, 3)
        inputs = jr.uniform(input_subkey, minval=0.0, maxval=1.0, shape=(batch_size, 4))
        labels = jr.randint(label_subkey, shape=(batch_size, 1), minval=0, maxval=9)
        yield TrainingBatch(inputs=inputs, labels=labels)


@eqx.filter_jit
@eqx.filter_grad
def loss_fn(model, batch, state=None):
    logits = jax.vmap(model)(batch["inputs"].reshape(batch["inputs"].shape[0], 4))
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["labels"].reshape(-1)
    ).mean()
    return loss


model_factory = ModelFactory(
    constructor=MLP,
    base_kwargs={"in_size": 4, "out_size": 1, "width_size": 16, "depth": 3},
    width_kwargs_names=("width_size",),
)
optimizer_factory = OptimizerFactory(
    optimizer_fn=optax.adam,
    hyperparams={"learning_rate": 1e-3},
)
training_cfg = TrainingConfig(
    model_factory=model_factory,
    optimizer_factory=optimizer_factory,
    loss_fn=loss_fn,
    width_multiplier=4.0,
)

# Coordinate check
coord_cfg = CoordinateCheckConfig(
    widths=[2**i for i in range(-1, 9)],
    rng_seeds=range(4),
    dataset_factory=lambda: make_dataset(batch_size=128, rng_seed=0),
    steps=50,
)
runner = CoordinateCheckRunner(training_cfg, coord_cfg)
runner.run(output_dir="results/coord_check")
