import jax
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import optax
from jax.tree_util import GetAttrKey, SequenceKey, DictKey
import jax.tree as jt
# Add src to path to import mup_equinox
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from mup_equinox.utils import get_value_at_path
from mup_equinox.config import ModelFactory
from mup_equinox.scalers import scale_gradients
from ssm import SSMDecoder, ContinuousSSMLayer

def _get_node(root, path):
    node = root
    for key in path:
        if isinstance(key, GetAttrKey): node = getattr(node, key.name)
        elif isinstance(key, SequenceKey): node = node[key.idx]
        elif isinstance(key, DictKey): node = node[key.key]
    return node

def path_to_string(path):
    """Converts a JAX tree path to a string representation."""
    segments = []
    for key in path:
        if isinstance(key, GetAttrKey):
            segments.append(f".{key.name}")
        elif isinstance(key, SequenceKey):
            segments.append(f"[{key.idx}]")
        elif isinstance(key, DictKey):
            segments.append(f"['{key.key}']")
        else:
            segments.append(f"[{key}]")
    return "".join(segments).lstrip(".")



def compute_empirical_scaling_laws(model_fn, widths, param_type="standard", target_layers='all'):
    """
    Loops through all parameters to find their ideal scaling law exponent.
    """
    results = []

    print(f"Sweeping widths: {widths} with param_type: {param_type}...")
    
    
    for width in widths:
        print(f"  Processing width {width}...")
        model, state, metadata = model_fn(width, param_type=param_type)
        
        batch_size = 25
        seq_len = 10
        input_dim = 5
        x_in = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, input_dim)) 
        
        if width == widths[0]:
            # Hardcoded mapping based on parameter_dependencies.csv
            param_to_layer = {
                "encoder.weight": "post_encoder",
                "encoder.bias": "post_encoder",
                "ssm_blocks[0].ssm.Lambda_re": "ssm_state_0",
                "ssm_blocks[0].ssm.Lambda_im": "ssm_state_0",
                "ssm_blocks[0].ssm.B": "ssm_state_0",
                "ssm_blocks[0].ssm.C": "ssm_output_0",
                "ssm_blocks[0].ssm.D": "ssm_output_0",
                "ssm_blocks[0].ssm.log_step": "ssm_state_0",
                "ssm_blocks[0].glu.w1.weight": "ssm_post_glu_0",
                "ssm_blocks[0].glu.w1.bias": "ssm_post_glu_0",
                "ssm_blocks[0].glu.w2.weight": "ssm_post_glu_0",
                "ssm_blocks[0].glu.w2.bias": "ssm_post_glu_0",
                "decoder.weight": "output",
                "decoder.bias": "output"
            }
        
        def get_activations_and_loss(m):
            # Capture activations
            # We use the initial state as requested
            acts = jax.vmap(m.get_activations, axis_name="batch", in_axes=(0, None, None))(x_in, state, target_layers)
            
            # Compute EMA of input as target
            # y_t = alpha * x_t + (1-alpha) * y_{t-1}
            alpha = 0.9
            def ema_step(carry, x_t):
                y_t = alpha * x_t + (1 - alpha) * carry
                return y_t, y_t
            
            _, ema_target = jax.vmap(lambda x: jax.lax.scan(ema_step, jnp.zeros(input_dim), x))(x_in)
            
            # MSE Loss
            loss = jnp.mean((acts["output"] - ema_target) ** 2)
            return loss, acts

        # Get global gradients
        (loss, activations), global_grads = eqx.filter_value_and_grad(get_activations_and_loss, has_aux=True)(model)

        scaler = scale_gradients(metadata, optimizer_type="sgd_like", param_type=param_type)
        scaled_grads, _ = scaler.update(global_grads, optax.EmptyState(), model)
        

        def probe_parameter(path, param, param_grad):
            if param_grad is None: 
                return 
            if not eqx.is_inexact_array(param):
                return

            param_name = path_to_string(path)

            
            target_key = param_to_layer.get(param_name, "output")
            target_act = activations[target_key]
            norm_x = jnp.mean(jnp.linalg.norm(target_act, axis=-1))
            
            
            # Compute JVP of the activation w.r.t this parameter
            def forward_slice_for_leaf(p_leaf):
                model_new = eqx.tree_at(lambda m: get_value_at_path(m, path), model, p_leaf)
                new_acts = jax.vmap(model_new.get_activations, axis_name="batch",  in_axes=(0, None, None))(x_in, state, 'all')
                return new_acts[target_key]

            _, tangent_out = jax.jvp(
                forward_slice_for_leaf, 
                (param,), 
                (param_grad,)
            )
            
            norm_dx = jnp.mean(jnp.linalg.norm(tangent_out, axis=-1))
            ideal_eta = float(norm_x) / (float(norm_dx) + 1e-9)
            
            results.append({
                "width": width,
                "param_name": param_name,
                "target_layer": target_key,
                "norm_x": float(norm_x),
                "norm_dx": float(norm_dx),
                "ideal_eta": ideal_eta
            })

        jt.map_with_path(probe_parameter, model, scaled_grads)

    return pd.DataFrame(results)

def plot_results(df, output_pdf="scaling_laws.pdf"):
    """
    Plots ideal_eta vs width for each parameter, grouped by target layer.
    """
    df = df[df["norm_dx"] > 1e-6]
    
    target_layers = df["target_layer"].unique()
    
    # Setup PDF
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(output_pdf) as pdf:
        for layer in target_layers:
            layer_df = df[df["target_layer"] == layer]
            if layer_df.empty: continue
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=layer_df, x="width", y="ideal_eta", hue="param_name", marker="o")
            
            plt.xscale("log", base=2)
            plt.yscale("log", base=2)
            plt.title(f"Ideal Scaling for Layer: {layer}")
            plt.xlabel("Width")
            plt.ylabel("Ideal Eta (||x|| / ||dx||)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            
            pdf.savefig()
            plt.close()
            
    print(f"Plots saved to {output_pdf}")

def summarize_scaling_factors(df):
    """
    Computes the slope of log(ideal_eta) vs log(width) for each parameter.
    """
    summary = []
    
    for param_name, group in df.groupby("param_name"):
        if len(group) < 2: continue
        
        log_width = np.log(group["width"])
        log_eta = np.log(group["ideal_eta"])
        
        slope, intercept = np.polyfit(log_width, log_eta, 1)
        
        summary.append({
            "param_name": param_name,
            "scaling_exponent": slope,
            "intercept": intercept
        })
        
    return pd.DataFrame(summary).sort_values("param_name")

if __name__ == "__main__":
    def model_fn(width, param_type="standard"):
        factory = ModelFactory(
            constructor=SSMDecoder,
            base_kwargs={
                "input_dim": 5,
                "output_dim": 5,
                "dim_ssm_io": 2,
                "dim_ssm_state": 2,
                "num_ssm_layers": 1,
                "ssm_layer_cls": ContinuousSSMLayer,
                "ssm_layer_kwargs": {
                    "dt_min": 0.1,
                    "dt_max": 0.1,
                }
            },
            width_kwargs_names=("dim_ssm_io", "dim_ssm_state"),
            param_type=param_type
        )
        return factory.build(width)

    widths = [64, 128, 256, 512, 1024, 2048]
    
    print("Starting empirical scaling analysis...")
    df = compute_empirical_scaling_laws(model_fn, widths, param_type="muP_SSM")
    
    os.makedirs("results/empirical_scaling_laws", exist_ok=True)
    df.to_csv("results/empirical_scaling_laws/raw_results.csv", index=False)
    print("Saved raw_results.csv")
    
    summary = summarize_scaling_factors(df)
    print("\nEmpirical Scaling Factors (Slope of log(eta) vs log(width)):")
    print(summary)
    summary.to_csv("results/empirical_scaling_laws/scaling_summary.csv", index=False)
    
    plot_results(df, "results/empirical_scaling_laws/scaling_laws.pdf")