from mup_equinox import (
    TrainingConfig, 
    CoordinateCheckConfig, 
    ModelFactory, 
    OptimizerFactory,
    CoordinateCheckRunner)
from mup_equinox.coord_check import run_coordinate_checks
import equinox as eqx
import optax
from typing import Iterator, TypedDict
from jaxtyping import Array, Float, Integer
import jax.random as jr
import jax
from ssm import SSMDecoder
from functools import partial


def make_nbit_memory_task_loaders(batch_size: int = 128, rng_seed: int = 0, n_coarse_steps: int = 25, upsampling_rate: int = 4, n_bits: int = 10, p_ticks: float = 0.1) -> tuple[Iterator, Iterator]:
    from nbit_memory_task import nbatch_nbit_memory_dataloader
    key = jr.PRNGKey(rng_seed)
    key, train_key, test_key = jr.split(key, 3)
    train_loader = nbatch_nbit_memory_dataloader(train_key, batch_size, n_coarse_steps, upsampling_rate, n_bits, p_ticks)
    test_loader = nbatch_nbit_memory_dataloader(test_key, batch_size, n_coarse_steps, upsampling_rate, n_bits, p_ticks)
    return train_loader, test_loader
#######


@eqx.filter_jit
@eqx.filter_grad
def loss_fn(model, batch, state=None) -> tuple[Float[Array, ""], eqx.nn.State]:
    inputs, labels = batch
    preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, None))(inputs, state, jr.PRNGKey(0))
    loss = optax.squared_error(
        preds, labels
    ).mean()
    return loss


model_factory = ModelFactory(
    constructor=SSMDecoder, 
    base_kwargs={
        "input_dim": 2,
        "output_dim": 2,
        "dim_ssm_io": 8,
        "dim_ssm_state": 8,
        "num_ssm_layers": 1,
        "ssm_layer_cls": 'ContinuousSSMLayer',
        "ssm_layer_kwargs":
            {   
                "dt_min": 0.001,
                "dt_max": 0.01,
                "a_initialisation": "s4d_inv",
                "rand_real": False,
                "rand_imag": False
            }
        },
    width_kwargs_names=("dim_ssm_io", "dim_ssm_state"),
    )

optimizer_factories = {
    "sgd": OptimizerFactory(optimizer_fn=optax.sgd, hyperparams={"learning_rate": 1e-2}),
    "adam": OptimizerFactory(optimizer_fn=optax.adam, hyperparams={"learning_rate": 1e-3}),
}

training_cfg = TrainingConfig(
    model_factory=model_factory, 
    optimizer_factory=optimizer_factories["sgd"], 
    dataset_factory = partial(make_nbit_memory_task_loaders,
        batch_size=128, n_coarse_steps=3, upsampling_rate=3, n_bits=2, p_ticks=0.2
    ),
    loss_fn=loss_fn,
    width_multiplier=4.0
)

# Coordinate check
coord_cfg = CoordinateCheckConfig(
    widths          = tuple(2**i for i in range(1, 11)),
    num_repetitions = 10,
    steps           = (1,100,1000),
    param_types     = ("muP_SSM","muP_3","standard")
)

if __name__ == "__main__":
    run_coordinate_checks(
        training_cfg,
        coord_cfg,
        optimizer_factories,
        output_base_dir='examples/ssm/results'
    )