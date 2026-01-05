import jax
import jax.numpy as jnp
import jax.random as jr
import functools
from jaxtyping import Array

@functools.partial(jax.jit, static_argnames=("batch_size", "seq_len", "input_dim", "output_dim"))
def build_white_noise_batch(batch_size, seq_len, input_dim, output_dim, key):
    """
    Generate white noise inputs with identity autocorrelation.
    """
    key, input_key, target_key = jr.split(key, 3)
    # Standard normal distribution for white noise
    inputs = jr.normal(input_key, (batch_size, seq_len, input_dim))
    # Random targets
    targets = jr.normal(target_key, (batch_size, seq_len, output_dim))
    return inputs, targets

def white_noise_dataloader(key: Array, batch_size: int = 16, seq_len: int = 100, input_dim: int = 1, output_dim: int = 1):
    while True:
        key, subkey = jr.split(key)
        inputs, targets = build_white_noise_batch(
            batch_size, seq_len, input_dim, output_dim, subkey
        )
        yield inputs, targets
