import equinox as eqx
import jax.random as jr
import pytest

import mup_equinox as mup


class DummyCfg:
    rng_seed = 0
    args = {"in_size": 2, "out_size": 2, "width_size": 3, "depth": 2}
    width_args = ["width_size"]


@pytest.fixture
def key():
    return jr.PRNGKey(0)


def test_build_base_and_scaled_models_width_multiplier():
    base_cfg = DummyCfg()
    base_model, scaled_model, _ = mup.build_base_and_scaled_models(
        eqx.nn.MLP, base_cfg, width_multiplier=2.0
    )
    # hidden width doubled
    assert scaled_model.width_size == base_model.width_size * 2
    # output dim unchanged
    assert scaled_model.out_size == base_model.out_size
