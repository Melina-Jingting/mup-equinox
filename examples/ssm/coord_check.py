from mup_equinox import (
    TrainingConfig, 
    CoordinateCheckConfig, 
    ModelFactory, 
    OptimizerFactory,
    CoordinateCheckRunner)
import equinox as eqx
import optax
from typing import Iterator, TypedDict
from jaxtyping import Array, Float, Integer
import jax.random as jr
import jax
from s5 import DeepSSM



###### Dataset 
# class TrainingBatch(TypedDict):
#     inputs: Float[Array, "batch timesteps channels"]
#     labels: Integer[Array, "batch timesteps channels"]
    
# def make_dataset(batch_size: int = 32, rng_seed: int = 0, timesteps: int = 100, in_channels: int = 10, out_channels: int = 2) -> Iterator[TrainingBatch]:
#     key = jr.PRNGKey(rng_seed)
#     while True:
#         key, input_subkey, label_subkey = jr.split(key, 3)
#         inputs = jr.uniform(
#             input_subkey, minval=0.0, maxval=1.0, shape=(batch_size, timesteps, in_channels)
#         )
#         labels = jr.uniform(
#             label_subkey, minval=0.0, maxval=1.0, shape=(batch_size, timesteps, out_channels)
#         )
#         yield TrainingBatch(inputs=inputs, labels=labels)

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
    preds, state = jax.vmap(model, in_axes=(0, None, None))(inputs, state, jr.PRNGKey(0))
    loss = optax.squared_error(
        preds, labels
    ).mean()
    return loss


model_factory = ModelFactory(
    constructor=DeepSSM, 
    base_kwargs={
        "in_dim": 2,
        "ssm_io_dim": 8,
        "ssm_state_dim": 8,
        "out_dim": 2,
        "ssm_num_layers": 1
        },
    width_kwargs_names=("ssm_io_dim", "ssm_state_dim"),
    )
optimizer_factory = OptimizerFactory(
    optimizer_fn=optax.adam,
    hyperparams={"learning_rate": 1e-2},
)
training_cfg = TrainingConfig(
    model_factory=model_factory, 
    optimizer_factory=optimizer_factory, 
    loss_fn=loss_fn,
    width_multiplier=4.0
)

# Coordinate check
coord_cfg = CoordinateCheckConfig(
    widths = [2**i for i in range(0, 9)],
    rng_seeds = range(2),
    dataset_factory = lambda: make_nbit_memory_task_loaders(
        batch_size=128, rng_seed=0, n_coarse_steps=5, upsampling_rate=3, n_bits=2, p_ticks=0.2
    ),
    steps  = 200,
    param_types = ["muP_SSM","muP_3","standard"]
)
runner = CoordinateCheckRunner(training_cfg, coord_cfg)
runner.run(output_dir='examples/ssm/results/adam_3')
