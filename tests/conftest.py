"""Fixtures. Everything here is deliberately tiny.

None of these tests check that the GAN learns anything - five epochs on
10,000 images is the experiment, not the test suite. They check that the
mechanics are what the report says they are: which parameters move, which
gradients flow, which numbers are reproducible.
"""

import pytest
import torch
from torch import nn

from gan_mnist import RunConfig, models


@pytest.fixture
def config():
    """A run small enough to execute in a test, identical in structure."""
    return RunConfig(
        name="test", subset_size=256, batch_size=8, epochs=1,
        checkpoints=(0, 1), n_eval_samples=4,
    )


@pytest.fixture
def device():
    # CPU regardless of what the machine offers: these tests are about
    # gradient bookkeeping, and CPU is where a failure is easiest to read.
    return torch.device("cpu")


@pytest.fixture
def nets(config, device):
    generator, discriminator = models.build(config)
    return generator.to(device), discriminator.to(device)


@pytest.fixture
def optimizers(nets, config):
    generator, discriminator = nets
    return (
        torch.optim.Adam(generator.parameters(), lr=config.generator_lr),
        torch.optim.Adam(discriminator.parameters(), lr=config.discriminator_lr),
    )


@pytest.fixture
def criterion():
    return nn.BCEWithLogitsLoss()


@pytest.fixture
def real_batch(config, device):
    rng = torch.Generator().manual_seed(0)
    return torch.rand(config.batch_size, 1, 28, 28, generator=rng).to(device) * 2 - 1


@pytest.fixture
def noise_batch(config, device):
    rng = torch.Generator().manual_seed(1)
    return torch.randn(config.batch_size, config.latent_dim, generator=rng).to(device)


def snapshot(module):
    """Clone every parameter, so a later comparison is against real values."""
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def unchanged(module, before) -> list:
    """Names of parameters that moved. Empty list means nothing moved."""
    moved = []
    for name, parameter in module.named_parameters():
        if not torch.equal(parameter.detach(), before[name]):
            moved.append(name)
    return moved
