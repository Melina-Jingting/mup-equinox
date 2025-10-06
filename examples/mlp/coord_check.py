from mup_equinox import (
    TrainingConfig,
    CoordinateCheckConfig,
    ModelFactory,
    OptimizerFactory,
    CoordinateCheckRunner,
)
import equinox as eqx
import optax
import jax
from mlp import MLP
from typing import Iterator


def make_tfds_dataloader(batch_size: int = 128, rng_seed: int = 0, dataset: str = "mnist") -> tuple[Iterator, Iterator]: 
    """
    Set up train and test loaders using TensorFlow Datasets. 
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.experimental.set_visible_devices([], "GPU")

    (train_loader, test_loader), info = tfds.load(
        dataset,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    def _transform(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, [-1])  # flatten
        return image, label
    
    train_loader_augmented = train_loader.map(_transform)
    train_loader_batched = train_loader_augmented.shuffle(
        buffer_size=10_000, reshuffle_each_iteration=True
    ).batch(batch_size, drop_remainder=True)

    test_loader_batched = test_loader.batch(batch_size, drop_remainder=True)
    return train_loader_batched.as_numpy_iterator(), test_loader_batched.as_numpy_iterator()
    


@eqx.filter_jit
@eqx.filter_grad
def loss_fn(model, batch, state=None):
    inputs, labels = batch
    logits = jax.vmap(model)(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, labels
    ).mean()
    return loss


model_factory = ModelFactory(
    constructor=MLP,
    base_kwargs={"in_dim": 784, "out_dim": 1, "width": 16, "depth": 3}, #28x28
    width_kwargs_names=("width",),
)
optimizer_factory = OptimizerFactory(
    optimizer_fn=optax.adam,
    hyperparams={"learning_rate": 1e-2},
)
training_cfg = TrainingConfig(
    model_factory=model_factory,
    optimizer_factory=optimizer_factory,
    loss_fn=loss_fn,
    width_multiplier=4.0,
)

# Coordinate check
coord_cfg = CoordinateCheckConfig(
    widths=[2**i for i in range(1, 9)],
    rng_seeds=range(4),
    dataset_factory=lambda: make_tfds_dataloader(batch_size=128, rng_seed=0),
    steps=10,
)
runner = CoordinateCheckRunner(training_cfg, coord_cfg)
runner.run(output_dir="examples/mlp/results/adam")
