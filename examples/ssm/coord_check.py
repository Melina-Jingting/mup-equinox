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
import jax.numpy as jnp
from ssm import SSMDecoder
from functools import partial
import os
import argparse
import sys

# Enable JAX compilation cache to mitigate slow compile times and timeouts
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.jax_cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)


def make_nbit_memory_task_loaders(batch_size: int = 16, rng_seed: int = 0, n_coarse_steps: int = 25, upsampling_rate: int = 4, n_bits: int = 10, p_ticks: float = 0.1) -> tuple[Iterator, Iterator]:
    from nbit_memory_task import nbatch_nbit_memory_dataloader
    key = jr.PRNGKey(rng_seed)
    key, train_key, test_key = jr.split(key, 3)
    train_loader = nbatch_nbit_memory_dataloader(train_key, batch_size, n_coarse_steps, upsampling_rate, n_bits, p_ticks)
    test_loader = nbatch_nbit_memory_dataloader(test_key, batch_size, n_coarse_steps, upsampling_rate, n_bits, p_ticks)
    return train_loader, test_loader

def make_white_noise_task_loaders(batch_size: int = 16, rng_seed: int = 0, seq_len: int = 200, input_dim: int = 1, output_dim: int = 1) -> tuple[Iterator, Iterator]:
    from white_noise_task import white_noise_dataloader
    key = jr.PRNGKey(rng_seed)
    key, train_key, test_key = jr.split(key, 3)
    train_loader = white_noise_dataloader(train_key, batch_size, seq_len, input_dim, output_dim)
    test_loader = white_noise_dataloader(test_key, batch_size, seq_len, input_dim, output_dim)
    return train_loader, test_loader
#######


@eqx.filter_jit
@eqx.filter_grad
def loss_fn(model, batch, state=None) -> tuple[Float[Array, ""], eqx.nn.State]:
    inputs, labels = batch
    preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, None))(inputs, state, jr.PRNGKey(0))
    
    # Use squared absolute difference for complex support
    # optax.squared_error computes (x-y)^2 which is invalid for minimizing distance in complex plane
    diff = preds - labels
    loss = jnp.mean(jnp.abs(diff) ** 2)
    
    return loss


import argparse
import sys

# ... existing imports ...

if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    # Parse custom arguments
    parser = argparse.ArgumentParser(description="SSM Coordinate Check", add_help=False)
    parser.add_argument("--tau", type=float, default=0.01, help="dt_min and dt_max value")
    parser.add_argument("--upsampling_rate", type=int, default=1, help="Upsampling rate")
    parser.add_argument("--n_coarse_steps", type=int, default=5, help="Number of coarse steps")
    
    # Parse known args to avoid conflict with run_coordinate_checks' parser
    args, remaining_argv = parser.parse_known_args()
    
    # Update sys.argv so run_coordinate_checks doesn't see our custom args
    sys.argv = [sys.argv[0]] + remaining_argv

    tau = args.tau
    upsampling_rate = args.upsampling_rate
    n_coarse_steps = args.n_coarse_steps

    model_factory = ModelFactory(
        constructor=SSMDecoder, 
        base_kwargs={
            "input_dim": 1,
            "output_dim": 1,
            "dim_ssm_io": 2,
            "dim_ssm_state": 2,
            "num_ssm_layers": 1,
            "ssm_layer_cls": 'ContinuousSSMLayer',
            "ssm_layer_kwargs":
                {   
                    "dt_min": tau,
                    "dt_max": tau,
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
            batch_size=32, n_coarse_steps=n_coarse_steps, upsampling_rate=upsampling_rate, n_bits=1, p_ticks=0.3
        ),
        loss_fn=loss_fn,
        width_multiplier=4.0
    )

    # Coordinate check
    coord_cfg = CoordinateCheckConfig(
        widths          = tuple(2**i for i in range(1, 14)),
        num_repetitions = 10,
        steps           = (1,10, 100, 10000),
        param_types     = ("muP_SSM","muP_SSM_Lambda_scaled","standard") 
    )
    
    # Use absolute path to ensure results are saved where expected
    output_dir = os.path.abspath(f'examples/ssm/results/tau_{tau}_ups_{upsampling_rate}_steps_{n_coarse_steps}/')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting coordinate checks. Results will be saved to: {output_dir}")
    
    run_coordinate_checks(
        training_cfg,
        coord_cfg,
        optimizer_factories,
        output_base_dir=output_dir
    )