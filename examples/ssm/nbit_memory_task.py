# Equinox and JAX-related imports
import jax
import jax.numpy as jnp
import jax.random as jr
import functools
from jaxtyping import Array

@functools.partial(jax.jit, static_argnames=("batch_size","n_coarse_steps","upsampling_rate","n_bits","p_ticks"))
def build_nbatch_nbit_memory(batch_size, n_coarse_steps, upsampling_rate, n_bits, p_ticks, key):
    """
    Generate n-bit memory tasks with upsampled input and targets.

    Parameters:
        batch_size (int): Number of samples in the batch.
        n_coarse_steps (int): Number of coarse time steps.
        upsampling_rate (int): Upsampling factor for input and targets.
        n_bits (int): Number of bits in the input sequence.
        p_ticks (float): Probability of a tick at each time step.
        key (jax.random.PRNGKey): Random key for reproducibility.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple of (input, target), both as JAX arrays.
    """
    key, ticks_key, flips_key = jax.random.split(key, 3)
    ticks = jax.random.bernoulli(ticks_key, p_ticks, shape=(batch_size, n_coarse_steps, n_bits))
    flips = jax.random.bernoulli(flips_key, 0.5, shape=(batch_size, n_coarse_steps, n_bits)) * 2 - 1
    input = jnp.multiply(ticks, flips)
    def scan_last_flip(seq_1d):
        def body(carry, x):
            next_carry = jnp.where(x != 0, x, carry)
            return next_carry, next_carry
        _, out = jax.lax.scan(body, jnp.array(0., dtype=seq_1d.dtype), seq_1d)
        return out
    target = jax.vmap(
            lambda x: jax.vmap(scan_last_flip, in_axes=(1), out_axes=(1))(x),
        in_axes=(0))(input)

    input = jnp.repeat(input, repeats=upsampling_rate, axis=1)
    target = jnp.repeat(target, repeats=upsampling_rate, axis=1)
    return input, target

def nbatch_nbit_memory_dataloader(key: Array, batch_size: int = 128, n_coarse_steps: int = 25, upsampling_rate: int = 10, n_bits: int = 3, p_ticks: float = 0.33):
    while True:
        key, subkey = jax.random.split(key)
        inputs, targets = build_nbatch_nbit_memory(
            batch_size, n_coarse_steps, upsampling_rate, n_bits, p_ticks, key
        )
        yield inputs, targets
