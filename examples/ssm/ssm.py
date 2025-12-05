from typing import Union, Any, Dict

import jax
import equinox as eqx

from jax import random as jr
import jax.numpy as jnp
import numpy as np
from jax.nn.initializers import lecun_normal, normal
from typing import List, Optional



def init_s4d_inv(N):
    return -0.5 + 1j * N/np.pi * (N / (2* jnp.arange(N) + 1) - 1)

def init_s4d_lin(N):
    return -0.5 + 1j * np.pi * np.arange(N)

def init_s4d_real(N):
    return -( jnp.arange(N) + 1 )
    

def init_lambda(method: str, N, rand_real=False, rand_imag=False, rand_key=None):    
    if method == "s4d_inv":
        Lambda = init_s4d_inv(N)
    elif method == "s4d_lin":
        Lambda = init_s4d_lin(N)
    elif method == "s4d_real":
        Lambda = init_s4d_real(N)
    else:
        raise NotImplementedError(f"Unknown initialization method: {method}")
    
    # Randomisation schemes from Gu et al., http://arxiv.org/abs/2206.11893
    # Real components ~ U(-1, 0) instead of fixed -0.5
    # Imaginary components ~ U(0, pi * N) instead of fixed spacing, based on S4D-Lin
    if (rand_real or rand_imag) and rand_key is None:
        raise ValueError("rand_key must be provided if rand_real or rand_imag is True")
    if rand_real:
        real_key, rand_key = jr.split(rand_key)
        Lambda = Lambda.at[:].set(-jr.uniform(real_key, Lambda.real.shape) + Lambda.imag * 1j)
    if rand_imag:
        imag_key, rand_key = jr.split(rand_key)
        Lambda = Lambda.at[:].set(Lambda.real + np.pi * N * (jr.uniform(imag_key, Lambda.imag.shape)) * 1j)

    return Lambda    
    

def init_log_steps(key, N, dt_min, dt_max):
    """Initialize an array of learnable timescale parameters
    Args:
        key: jax jr key
        input: tuple containing the array shape H and
               dt_min and dt_max
    Returns:
        initialized array of timescales (float32): (H,)
    """
    return jr.uniform(key, (N,)) * (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)

def discretise_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretisation step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretise_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretisation step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar

def discretise(method, Lambda, B_tilde, Delta):
    if method in ["zoh"]:
        Lambda_bar, B_bar = discretise_zoh(Lambda, B_tilde, Delta)
    elif method in ["bilinear"]:
        Lambda_bar, B_bar = discretise_bilinear(Lambda, B_tilde, Delta)
    else:
        raise NotImplementedError(
            "Discretization method {} not implemented".format(method)
        )
    return Lambda_bar, B_bar


@jax.vmap
def linear_recurrence_op(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def compute_hidden_states(Lambda_bar, B_bar, input_sequence):
    """Compute the LxH output of discretized SSM given an LxH input.
    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        input_sequence (float32): input sequence of features         (L, H)
        conj_sym (bool):         whether conjugate symmetry is enforced
    Returns:
        ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * jnp.ones(
        (input_sequence.shape[0], Lambda_bar.shape[0])
    )
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(linear_recurrence_op, (Lambda_elements, Bu_elements))
    return xs


class ActivationCapturer:
    def __init__(self, layer_keys: Union[List[str], str, None]):
        self.layer_keys = layer_keys
        self.activations = {}

    def capture(self, key: str, value: Any):
        if self.layer_keys is None or self.layer_keys == "all" or key in self.layer_keys:
            self.activations[key] = value

    def merge(self, other_activations: Dict[str, Any], suffix: str = ""):
        for k, v in other_activations.items():
            self.activations[f"{k}{suffix}"] = v


class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))
    
class ContinuousSSMLayer(eqx.Module):
    Lambda_re: jax.Array # (P/2,) or (P,)
    Lambda_im: Optional[jax.Array]
    B: jax.Array # (P/2, H, 2) or (P, H)
    C: jax.Array # (H, P/2, 2) or (H, P)
    D: jax.Array # (H,)
    log_step: jax.Array # (P/2,) or (P,)

    dim_ssm_io: int
    dim_ssm_state: int
    conj_sym: bool = True
    discretisation: str = "zoh"
    a_initialisation: str = "s4d_inv"
    step_rescale: float = 1.0

    def __init__(
        self,
        *,
        dim_ssm_state,
        dim_ssm_io,
        dt_min,
        dt_max,
        a_initialisation = "s4d_inv",
        discretisation = "zoh",
        conj_sym = True,
        rand_real = False,
        rand_imag = False,
        key,
    ):
        A_key, B_key, C_key, D_key, step_key, key = jr.split(key, 6)

        self.dim_ssm_io = dim_ssm_io
        self.dim_ssm_state = dim_ssm_state
        if a_initialisation == 's4d_real': 
            self.conj_sym = False # S4D Real has no imaginary parts
        else:
            self.conj_sym = conj_sym

        if self.conj_sym:
            effective_dim_state = dim_ssm_state // 2
        else:
            effective_dim_state = dim_ssm_state
    
        Lambda = init_lambda(a_initialisation, effective_dim_state, rand_real, rand_imag, rand_key=A_key)
        self.Lambda_re = Lambda.real
        
        if self.conj_sym:
            self.Lambda_im = Lambda.imag
            self.B = lecun_normal(batch_axis=0)(B_key, (effective_dim_state, self.dim_ssm_io, 2)) / jnp.sqrt(2.0)
            self.C = lecun_normal(batch_axis=0)(C_key, (self.dim_ssm_io, effective_dim_state, 2)) / jnp.sqrt(2.0)
        else:
            self.Lambda_im = None
            self.B = lecun_normal(in_axis=-1, out_axis=-2)(B_key, (effective_dim_state, self.dim_ssm_io))
            self.C = lecun_normal(in_axis=-1, out_axis=-2)(C_key, (self.dim_ssm_io, effective_dim_state))

        self.D = normal(stddev=1.0)(D_key, (self.dim_ssm_io,))

        self.log_step = init_log_steps(step_key, effective_dim_state, dt_min, dt_max)

        self.discretisation = discretisation

    def _get_constants(self):
        if self.conj_sym:
            Lambda = self.Lambda_re + 1j * self.Lambda_im
            B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
            C_tilde = self.C[..., 0] + 1j * self.C[..., 1]
        else:
            Lambda = self.Lambda_re
            B_tilde = self.B
            C_tilde = self.C

        step = self.step_rescale * jnp.exp(self.log_step)
        return Lambda, B_tilde, C_tilde, step

    def __call__(self, input_sequence):
        Lambda, B_tilde, C_tilde, step = self._get_constants()
        Lambda_bar, B_bar = discretise(self.discretisation, Lambda, B_tilde, step)
        
        xs = compute_hidden_states(Lambda_bar, B_bar, input_sequence)
        
        if self.conj_sym:
            ys = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
        else:
            ys = jax.vmap(lambda x: (C_tilde @ x).real)(xs)
            
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du

    def get_activations(self, input_sequence, *, layer_keys: list[str] | str | None = None, return_outputs=False):
        capturer = ActivationCapturer(layer_keys)
        Lambda, B_tilde, C_tilde, step = self._get_constants()

        # Discretize
        Lambda_bar, B_bar = discretise(self.discretisation, Lambda, B_tilde, step)

        xs = compute_hidden_states(Lambda_bar, B_bar, input_sequence)
        capturer.capture("ssm_state", xs)
        
        if self.conj_sym:
            ys = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
        else:
            ys = jax.vmap(lambda x: (C_tilde @ x).real)(xs)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        output = ys + Du
        capturer.capture("ssm_output", output)
        if return_outputs:
            return capturer.activations, output
        else:
            return capturer.activations


SSM_LAYER_REGISTRY = {
    "ContinuousSSMLayer": ContinuousSSMLayer,
}

class SSMBlock(eqx.Module):
    norm: eqx.nn.BatchNorm
    ssm: eqx.Module
    glu: eqx.Module  # Changed from GLU to the more general eqx.Module
    drop: eqx.nn.Dropout

    def __init__(
        self,
        *,
        dim_ssm_io,
        dim_ssm_state,
        ssm_layer_cls,
        ssm_layer_kwargs,
        drop_rate=0.05,
        key,
    ):
        ssmkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=dim_ssm_io, axis_name="batch", channelwise_affine=False, mode="batch"
        )
        
        self.ssm = ssm_layer_cls(
            dim_ssm_io=dim_ssm_io,
            dim_ssm_state=dim_ssm_state,
            **ssm_layer_kwargs,
            key=ssmkey,
        )

        # Conditionally initialize the GLU or an Identity layer
        self.glu = GLU(dim_ssm_io, dim_ssm_io, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    # No changes are needed for the __call__ method.
    def __call__(self, x, state, *, key):
        """Compute S5 block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.ssm(x)
        x = jax.nn.gelu(x)
        x = self.drop(x, key=dropkey1)
        x = jax.vmap(self.glu)(x)  # This line works for both GLU and Identity
        x = self.drop(x, key=dropkey2)
        # x = skip + x
        return x, state

    def get_activations(self, x, state, layer_keys, *, return_outputs=False):
        """Compute S5 block."""
        capturer = ActivationCapturer(layer_keys)
        x, state = self.norm(x.T, state)
        x = x.T
        ssm_activations, ssm_y = self.ssm.get_activations(x, layer_keys=layer_keys, return_outputs=True)
        capturer.merge(ssm_activations)
        
        post_gelu = jax.nn.gelu(ssm_y)
        capturer.capture("ssm_post_gelu", post_gelu)
        
        post_glu = jax.vmap(self.glu)(post_gelu)
        capturer.capture("ssm_post_glu", post_glu)
        if return_outputs:
            return capturer.activations, post_glu, state
        else:
            return capturer.activations


class _MultiEncoder(eqx.Module):
    """Helper module to handle group-specific encoders."""
    encoders: List[eqx.nn.Linear]

    def __init__(self, input_dim, output_dim, num_groups, *, key):
        keys = jr.split(key, num_groups)
        self.encoders = [eqx.nn.Linear(input_dim, output_dim, key=k) for k in keys]

    def __call__(self, x, group_idx):
        # x shape: (Length, InputDim)
        # We vmap the encoders over the sequence length
        funcs = [jax.vmap(enc, in_axes=0, out_axes=0) for enc in self.encoders]
        return jax.lax.switch(group_idx, funcs, x)

class SSMDecoder(eqx.Module):
    encoder: Union[eqx.nn.Linear, _MultiEncoder]
    encoder_dropout: eqx.nn.Dropout
    ssm_blocks: List[SSMBlock]
    decoder: eqx.nn.Linear
    decoder_dropout: eqx.nn.Dropout
    
    # Metadata flags
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        key,
        *,
        input_dim,
        output_dim,
        dim_ssm_io=32,
        dim_ssm_state=32,
        num_ssm_layers=2,
        num_dataset_groups=1,
        dropout_p=0.1,
        ssm_layer_cls: Union[str, Any] = "S5Layer",
        ssm_layer_kwargs: Dict = {},
    ):
        # key = jr.PRNGKey(rng_seed)
        encoder_key, block_key, decoder_key = jr.split(key, 3)
        
        # Resolve SSM Layer Class
        if isinstance(ssm_layer_cls, str):
            if ssm_layer_cls not in SSM_LAYER_REGISTRY:
                raise ValueError(f"Unknown SSM layer class: {ssm_layer_cls}")
            ssm_layer_cls = SSM_LAYER_REGISTRY[ssm_layer_cls]

        # Initialize Encoder (Single or Multi-Group)
        if num_dataset_groups > 1:
            self.encoder = _MultiEncoder(
                input_dim, dim_ssm_io, num_dataset_groups, key=encoder_key
            )
        else:
            self.encoder = eqx.nn.Linear(input_dim, dim_ssm_io, key=encoder_key)
            
        self.encoder_dropout = eqx.nn.Dropout(p=dropout_p)

        block_keys = jr.split(block_key, num_ssm_layers)
        self.ssm_blocks = [
            SSMBlock(
                dim_ssm_io=dim_ssm_io,
                dim_ssm_state=dim_ssm_state,
                ssm_layer_cls=ssm_layer_cls,
                ssm_layer_kwargs=ssm_layer_kwargs,
                drop_rate=dropout_p,
                key=k,
            )
            for k in block_keys
        ]

        self.decoder = eqx.nn.Linear(dim_ssm_io, output_dim, key=decoder_key)
        self.decoder_dropout = eqx.nn.Dropout(p=dropout_p)

    def __call__(self, x, state, key, group_idx=None):
        """
        Compute SSM Decoder.
        Args:
            x: Input tensor.
            state: Model state.
            key: PRNG key.
            group_idx: Dataset group index (required if num_dataset_groups > 1).
        """
        if isinstance(self.encoder, _MultiEncoder):
            if group_idx is None:
                raise ValueError("group_idx must be provided for multi-group encoder")
            x = self.encoder(x, group_idx)
        else:
            x = jax.vmap(self.encoder)(x)

        num_dropout_layers = 2 + len(self.ssm_blocks)
        dropkeys = jr.split(key, num_dropout_layers)

        x = self.encoder_dropout(x, key=dropkeys[0])

        for block, k in zip(self.ssm_blocks, dropkeys[1:-1]):
            x, state = block(x, state, key=k)

        x = jax.vmap(self.decoder)(x)
        x = self.decoder_dropout(x, key=dropkeys[-1])
        return x, state

    def get_activations(self, x, state, layer_keys, group_idx=None):
        capturer = ActivationCapturer(layer_keys)

        if isinstance(self.encoder, _MultiEncoder):
            if group_idx is None:
                raise ValueError("group_idx must be provided for multi-group encoder")
            x = self.encoder(x, group_idx)
        else:
            x = jax.vmap(self.encoder)(x)
            
        capturer.capture("post_encoder", x)

        for i, block in enumerate(self.ssm_blocks):
            block_activations, x, state = block.get_activations(
                x, state, layer_keys=layer_keys, return_outputs=True
            )
            capturer.merge(block_activations, suffix=f"_{i}")

        x = jax.vmap(self.decoder)(x)
        capturer.capture("output", x)
        return capturer.activations