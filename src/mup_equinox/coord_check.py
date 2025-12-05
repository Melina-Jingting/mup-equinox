import jax.tree as jt
import jax.numpy as jnp
import pandas as pd
import dataclasses
from dataclasses import dataclass
from .config import TrainingConfig, OptimizerFactory
import equinox as eqx
from typing import Sequence, Callable, Iterable, Any 
import os
import numpy as np
import inspect
import matplotlib.lines as mlines
from .utils import ordered_tree_map
import argparse

@dataclass
class CoordinateCheckConfig:
    widths: Sequence[float]  # list or range
    num_repetitions: int  # handled in setup
    steps: Sequence[int] | int = (1, 10, 100)
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
        norms, deltas = [], []
        cfg = self.training_cfg
        
        rng_seeds = np.random.randint(0, 1e6, size=self.coord_cfg.num_repetitions)
        for param_type in self.coord_cfg.param_types:
            for width in self.coord_cfg.widths:
                for seed in rng_seeds:
                    train_loader, _ = cfg.dataset_factory(rng_seed=seed)
                    dataset_iter = iter(train_loader)
                    batch = next(dataset_iter)
                    sample_input_for_activation = batch[0][0]
                    
                    model, state, metadata = cfg.model_factory.with_rng(seed).with_param_type(param_type).build(width)
                    if not hasattr(model, "get_activations"):
                        raise AttributeError("Model must have method get_activations(x)->(activations)  for coordinate checks.")

                    optimizer = cfg.optimizer_factory.build(metadata)
                    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

                    a0 = self._get_activations(eqx.nn.inference_mode(model, value=True), sample_input_for_activation, state)
                    for s, nbatch in enumerate(dataset_iter):
                        grads = cfg.loss_fn(model, batch, state)
                        updates, opt_state = optimizer.update(grads, opt_state, model)
                        model = eqx.apply_updates(model, updates)
                        
                        if s + 1 in self.coord_cfg.steps:
                            a1 = self._get_activations(eqx.nn.inference_mode(model, value=True), sample_input_for_activation, state)
                            norm_a1 = {k: jnp.mean(jnp.abs(v)) for k, v in a1.items()}
                            norm_delta = {k: jnp.mean(jnp.abs(a1[k] - a0[k])) for k in a1.keys()}

                            norms.append({"param_type": param_type, "width_multiplier": width, "step": s + 1, "rng_seed": seed, **norm_a1})
                            deltas.append({"param_type": param_type, "width_multiplier": width, "step": s + 1, "rng_seed": seed, **norm_delta})

                        if s + 1 == max(self.coord_cfg.steps):
                            # No need to continue training beyond the max step
                            # Breaking early may cause a TensorFlow warning about the iterator 
                            # not being fully read. This is expected and can be ignored or suppressed.
                            break  

        self._save_results(norms, deltas, output_dir)

    def _save_results(self, norms, deltas, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for data, name in ((norms, "activation_norms"), (deltas, "activation_deltas")):
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
        
        self._plot_coord_check_results(data_dir=output_dir, metrics=self.coord_cfg.metrics, param_types=self.coord_cfg.param_types)
    
    def _plot_coord_check_results(
        self,
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

        # Load all dataframes first to determine steps
        dfs = {}
        all_steps = set()
        for metric in metrics:
            df = pd.read_csv(os.path.join(data_dir, f"{metric}.csv"))
            df["width_multiplier"] = np.log2(df["width_multiplier"])
            dfs[metric] = df
            if "step" in df.columns:
                all_steps.update(df["step"].unique())
        
        sorted_steps = sorted(list(all_steps)) if all_steps else [None]

        for step in sorted_steps:
            rows = len(metrics)
            cols = len(param_types)
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharey='row')
            axes = axes.reshape(rows, cols)

            layer_order: list[str] = []
            layer_colors: dict[str, Any] = {}
            layer_labels: dict[str, str] = {}

            for row_idx, metric in enumerate(metrics):
                df_full = dfs[metric]
                if step is not None:
                    df = df_full[df_full["step"] == step]
                else:
                    df = df_full

                layers = [
                    col
                    for col in df.columns
                    if col not in ("param_type", "width_multiplier", "rng_seed", "step")
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
                    if not df.empty:
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

            step_suffix = f"_step_{step}" if step is not None else ""
            plot_title = (title or "Coordinate Check Results") + (f" (Step {step})" if step is not None else "")
            fig.suptitle(plot_title, y=0.98)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            plt.savefig(os.path.join(data_dir, f"coordinate_check_plot{step_suffix}.png"))
            plt.close(fig)


def run_coordinate_checks(
    training_cfg: TrainingConfig,
    coord_cfg: CoordinateCheckConfig,
    optimizer_factories: dict[str, OptimizerFactory],
    output_base_dir: str,
    default_optimizers: list[str] = ["sgd"]
):

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=list(coord_cfg.steps))
    parser.add_argument("--name_prefix", type=str, default="")
    parser.add_argument("--optimiser", type=str, nargs="+", default=default_optimizers, choices=list(optimizer_factories.keys()))
    parser.add_argument("--num_repetitions", type=int, default=coord_cfg.num_repetitions)

    args = parser.parse_args()

    for opt_name in args.optimiser:
        opt_factory = optimizer_factories[opt_name]
        print(f"Running coordinate check: Optimizer={opt_name}, Steps={args.steps}")
        
        current_coord_cfg = dataclasses.replace(coord_cfg, steps=args.steps, num_repetitions=args.num_repetitions)
        current_training_cfg = dataclasses.replace(training_cfg, optimizer_factory=opt_factory)
        
        runner = CoordinateCheckRunner(current_training_cfg, current_coord_cfg)
        
        dir_name = f"{args.name_prefix}_{opt_name}"
        runner.run(output_dir=os.path.join(output_base_dir, dir_name))

