import jax.tree as jt
import jax.numpy as jnp
import pandas as pd
import dataclasses
from dataclasses import dataclass
from .config import TrainingConfig 
import equinox as eqx
from typing import Sequence, Callable, Iterable, Any 
import os
import numpy as np
# from .validators import ensure_activation_interface
# from ..training.builders import build_model_and_state, build_optimizer

@dataclass
class CoordinateCheckConfig:
    widths: Sequence[float]  # list or range
    rng_seeds: Sequence[int]  # handled in setup
    dataset_factory: Callable[[], Iterable]  # yields TrainingBatch
    steps: int = 100
    metrics: Sequence[str] = ("activation_norms", "activation_deltas")
    capture_layers: Sequence[str] | str = "all"

class CoordinateCheckRunner:
    def __init__(self, 
                 training_cfg: TrainingConfig,
                 coord_cfg: CoordinateCheckConfig):
        self.training_cfg = training_cfg
        self.coord_cfg = coord_cfg

    def run(self, output_dir):
        dataset_iter = self.coord_cfg.dataset_factory()
        batch = next(dataset_iter)
        inputs = batch["inputs"][0]

        norms, deltas = [], []
        for param_type in ["muP_3", "standard"]:
            for width in self.coord_cfg.widths:
                for seed in self.coord_cfg.rng_seeds:
                    cfg = dataclasses.replace(self.training_cfg, width_multiplier=width, rng_seed=seed)
                    model, state, metadata = cfg.model_factory.with_rng(seed).with_param_type(param_type).build(cfg.width_multiplier)
                    if not hasattr(model, "get_activations"):
                        raise AttributeError("Model must implement get_activations(...) for coordinate checks.")

                    optimizer = cfg.optimizer_factory.build(metadata)
                    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

                    a0 = eqx.nn.inference_mode(model, value=True).get_activations(inputs, layer_keys=self.coord_cfg.capture_layers)
                    for _ in range(self.coord_cfg.steps):
                        grads = cfg.loss_fn(model, batch)
                        updates, opt_state = optimizer.update(grads, opt_state, model)
                        model = eqx.apply_updates(model, updates)

                    a1 = eqx.nn.inference_mode(model, value=True).get_activations(inputs, layer_keys=self.coord_cfg.capture_layers)
                    norm_a1 = jt.map(lambda x: jnp.mean(jnp.abs(x)), a1)
                    norm_delta = jt.map(lambda x, y: jnp.mean(jnp.abs(x - y)), a0, a1)

                    norms.append({"param_type": param_type, "width_multiplier": width, "rng_seed": seed, **norm_a1})
                    deltas.append({"param_type": param_type, "width_multiplier": width, "rng_seed": seed, **norm_delta})

        self._save_results(norms, deltas, output_dir)

    def _save_results(self, norms, deltas, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for data, name in ((norms, "activation_norms"), (deltas, "activation_deltas")):
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
            
        plot_coord_check_results(data_dir=output_dir, metrics=self.coord_cfg.metrics)
        
            
def plot_coord_check_results(
    data_dir: str,
    metrics: Sequence[str] = ("activation_norms", "activation_deltas"),
    title: str | None = None,
):
    """Plot coordinate check results from CSV files in ``data_dir``."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")

    metric_labels = {
        "activation_norms": r"$\frac{\|x\|_2}{\sqrt{N_x}}$",
        "activation_deltas": r"$\frac{\|\Delta x\|_2}{\sqrt{N_x}}$",
    }
    param_labels = {"muP_3": r"$\mu P$", "standard": "SP"}

    rows = len(metrics)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows), sharey=False)
    axes = axes.reshape(rows, 2)

    legend_entries: dict[str, Any] = {}

    for row_idx, metric in enumerate(metrics):
        df = pd.read_csv(os.path.join(data_dir, f"{metric}.csv"))
        df["width_multiplier"] = np.log2(df["width_multiplier"])
        layers = [
            col
            for col in df.columns
            if col not in ("param_type", "width_multiplier", "rng_seed")
        ]

        for col_idx, param_type in enumerate(["muP_3", "standard"]):
            ax = axes[row_idx, col_idx]
            param_df = df[df["param_type"] == param_type]
            if param_df.empty:
                ax.axis("off")
                continue

            for layer in layers:
                line = sns.lineplot(
                    data=param_df,
                    x="width_multiplier",
                    y=layer,
                    marker="o",
                    label=layer,
                    legend=False,
                    ax=ax,
                )

            ax.set_title(param_labels.get(param_type, param_type)) if metric == metrics[0] else ax.set_title("")
            ax.set_ylabel(metric_labels.get(metric, metric.replace("_", " ").title())) if param_type == "muP_3" else ax.set_ylabel("")
            ax.set_xlabel(r"$\log_2 width$") if metric == metrics[-1] else ax.set_xlabel("")
            ax.set_xticks(range(int(np.min(df["width_multiplier"])), int(np.max(df["width_multiplier"]) + 1)))
        

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(labels), title="Layers")

    fig.suptitle(title or "Coordinate Check Results", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.savefig(os.path.join(data_dir, "coordinate_check_plot.png"))
    plt.close(fig)