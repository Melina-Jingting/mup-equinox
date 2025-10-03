from dataclasses import dataclass
import dataclasses
from omegaconf import OmegaConf
from typing import Callable, Sequence, Optional
import equinox as eqx
import optax
from .scalers import scale_initializations, scale_gradients
from .metadata import ParameterizationMetadata, build_param_metadata
import jax.numpy as jnp
import jax.random as jr


@dataclass
class ModelFactory:
    constructor: Callable[..., eqx.Module]
    base_kwargs: dict
    width_kwargs_names: Sequence[str]
    rng_seed: int = 0
    param_type: str = "muP_3"

    @classmethod
    def from_config(cls, constructor, cfg_path: str, **kwargs):
        cfg = OmegaConf.load(cfg_path)
        return cls(
            constructor=constructor,
            base_kwargs=dict(cfg.kwargs),
            width_kwargs_names=tuple(cfg.width_kwargs),
            rng_seed=cfg.get("rng_seed", 0),
            **kwargs,
        )

    def with_rng(self, rng_seed: int) -> "ModelFactory":
        return dataclasses.replace(self, rng_seed=rng_seed)

    def with_param_type(self, param_type: str) -> "ModelFactory":
        return dataclasses.replace(self, param_type=param_type)

    def _multiply_width(self, base_val, width_multiplier):
        if isinstance(base_val, int):
            return int(round(base_val * width_multiplier))
        return base_val * width_multiplier

    def _build_kwargs(self, width_multiplier: float) -> dict:
        width_kwargs = {
            name: self._multiply_width(self.base_kwargs[name], width_multiplier)
            for name in self.width_kwargs_names
        }
        return {**self.base_kwargs, **width_kwargs}

    def build(
        self, width_multiplier: float
    ) -> tuple[eqx.Module, eqx.nn.State, eqx.Module]:
        base_kwargs = self._build_kwargs(1.0)
        scaled_kwargs = self._build_kwargs(width_multiplier)

        base_model = self.constructor(**base_kwargs, key=jr.PRNGKey(self.rng_seed))
        model, state = eqx.nn.make_with_state(self.constructor)(
            **scaled_kwargs, key=jr.PRNGKey(self.rng_seed)
        )
        metadata = build_param_metadata(base_model, model)
        model = scale_initializations(model, metadata, param_type=self.param_type)
        return model, state, metadata


@dataclass
class OptimizerFactory:
    optimizer_fn: Callable[..., optax.GradientTransformation]
    hyperparams: dict
    param_type: str = "muP_3"

    @property
    def optimizer_type(self) -> str:
        name = self.optimizer_fn.__name__.lower()
        if "adam" in name or "lamb" in name or "adamw" in name:
            return "adam_like"
        elif "sgd" in name or "momentum" in name:
            return "sgd_like"
        else:
            raise ValueError(
                f"Cannot infer optimizer_type from optimizer_fn name '{self.optimizer_fn.__name__}'. Please use a standard optimizer."
            )

    def build(self, metadata) -> optax.GradientTransformation:
        base_opt = self.optimizer_fn(**self.hyperparams)
        scaled_opt = scale_gradients(
            metadata, optimizer_type=self.optimizer_type, param_type=self.param_type
        )
        return optax.chain(base_opt, scaled_opt)


@dataclass
class TrainingConfig:
    model_factory: ModelFactory
    optimizer_factory: OptimizerFactory
    loss_fn: Callable[[eqx.Module, dict, Optional[eqx.nn.State]], jnp.ndarray]  # (model, batch) -> loss
    width_multiplier: float
    rng_seed: int = 0
