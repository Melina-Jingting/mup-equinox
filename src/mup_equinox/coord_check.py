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
import inspect
import matplotlib.lines as mlines
from .utils import ordered_tree_map
# from .validators import ensure_activation_interface
# from ..training.builders import build_model_and_state, build_optimizer

@dataclass
class CoordinateCheckConfig:
    widths: Sequence[float]  # list or range
    rng_seeds: Sequence[int]  # handled in setup
    dataset_factory: Callable[[], Iterable]  # yields TrainingBatch
    steps: int = 100
    metrics: Sequence[str] = ("activation_norms", "activation_deltas")
    param_types: Sequence[str] = ("muP_3", "standard")
    capture_layers: Sequence[str] | str = "all"

class CoordinateCheckRunner:
    def __init__(self, 
                 training_cfg: TrainingConfig,
                 coord_cfg: CoordinateCheckConfig):
        self.training_cfg = training_cfg
        self.coord_cfg = coord_cfg

    def _get_activations(self, model, inputs, state=None):
        """Intelligently call get_activations based on the method signature."""
        sig = inspect.signature(model.get_activations)
        
        # Check if 'state' is in the signature
        if 'state' in sig.parameters:
            return model.get_activations(inputs, state=state, layer_keys=self.coord_cfg.capture_layers)
        else:
            return model.get_activations(inputs, layer_keys=self.coord_cfg.capture_layers)

    def run(self, output_dir):
        train_loader, _ = self.coord_cfg.dataset_factory()
        dataset_iter = iter(train_loader)
        batch = next(dataset_iter)
        sample_input_for_activation = batch[0][0]

        norms, deltas = [], []
        for param_type in self.coord_cfg.param_types:
            for width in self.coord_cfg.widths:
                for seed in self.coord_cfg.rng_seeds:
                    cfg = dataclasses.replace(self.training_cfg, width_multiplier=width, rng_seed=seed)
                    model, state, metadata = cfg.model_factory.with_rng(seed).with_param_type(param_type).build(cfg.width_multiplier)
                    if not hasattr(model, "get_activations"):
                        raise AttributeError("Model must have method get_activations(x)->(activations)  for coordinate checks.")

                    optimizer = cfg.optimizer_factory.build(metadata)
                    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

                    a0 = self._get_activations(eqx.nn.inference_mode(model, value=True), sample_input_for_activation, state)
                    for _ in range(self.coord_cfg.steps):
                        grads = cfg.loss_fn(model, batch, state)
                        updates, opt_state = optimizer.update(grads, opt_state, model)
                        model = eqx.apply_updates(model, updates)

                    a1 = self._get_activations(eqx.nn.inference_mode(model, value=True), sample_input_for_activation, state)
                    norm_a1 = {k: jnp.mean(jnp.abs(v)) for k, v in a1.items()}
                    norm_delta = {k: jnp.mean(jnp.abs(a1[k] - a0[k])) for k in a1.keys()}

                    norms.append({"param_type": param_type, "width_multiplier": width, "rng_seed": seed, **norm_a1})
                    deltas.append({"param_type": param_type, "width_multiplier": width, "rng_seed": seed, **norm_delta})

        self._save_results(norms, deltas, output_dir)

    def _save_results(self, norms, deltas, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for data, name in ((norms, "activation_norms"), (deltas, "activation_deltas")):
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
            
        plot_coord_check_results(data_dir=output_dir, metrics=self.coord_cfg.metrics, param_types=self.coord_cfg.param_types)
        
            
def plot_coord_check_results(
    data_dir: str,
    metrics: Sequence[str] = ("activation_norms", "activation_deltas"),
    param_types: Sequence[str] = ("muP_3", "standard"),
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
    param_labels = {"muP_3": r"$\mu P$", "muP_SSM": r"$\mu P-SSM$", "standard": "SP"}

    rows = len(metrics)
    cols = len(param_types)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharey='row')
    axes = axes.reshape(rows, cols)

    layer_order: list[str] = []
    layer_colors: dict[str, Any] = {}
    layer_labels: dict[str, str] = {}

    for row_idx, metric in enumerate(metrics):
        df = pd.read_csv(os.path.join(data_dir, f"{metric}.csv"))
        df["width_multiplier"] = np.log2(df["width_multiplier"])
        layers = [
            col
            for col in df.columns
            if col not in ("param_type", "width_multiplier", "rng_seed")
        ]

        if not layer_order and layers:
            layer_order = list(layers)
            palette = sns.color_palette("viridis", len(layer_order))
            layer_colors.update({layer: palette[idx] for idx, layer in enumerate(layer_order)})
            layer_labels.update({layer: f"{idx + 1}. {layer}" for idx, layer in enumerate(layer_order)})

        for col_idx, param_type in enumerate(param_types):
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
                    label=layer_labels.get(layer, layer),
                    legend=False,
                    ax=ax,
                    color=layer_colors.get(layer),
                )

            ax.set_title(param_labels.get(param_type, param_type)) if metric == metrics[0] else ax.set_title("")
            ax.set_ylabel(metric_labels.get(metric, metric.replace("_", " ").title())) if col_idx == 0 else ax.set_ylabel("")
            ax.set_xlabel(r"$\log_2 width\ multiplier$") if metric == metrics[-1] else ax.set_xlabel("")
            ax.set_xticks(range(int(np.min(df["width_multiplier"])), int(np.max(df["width_multiplier"]) + 1)))
            ax.set_yscale("log")

    if layer_order:
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                color=layer_colors[layer],
                marker="o",
                linestyle="-",
                label=layer_labels[layer],
            )
            for layer in layer_order
        ]
        fig.legend(
            legend_handles,
            [layer_labels[layer] for layer in layer_order],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=min(len(legend_handles), 5),
            title="Layers (in activation order)",
        )

    fig.suptitle(title or "Coordinate Check Results", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.savefig(os.path.join(data_dir, "coordinate_check_plot.png"))
    plt.close(fig)