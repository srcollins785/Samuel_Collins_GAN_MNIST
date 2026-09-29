"""Shapes, ranges and the logits/loss pairing."""

import torch
from torch import nn

from gan_mnist import models


def test_generator_maps_noise_to_image_shape(config):
    generator, _ = models.build(config)
    z = torch.randn(5, config.latent_dim)
    out = generator(z)
    assert out.shape == (5, 1, 28, 28)


def test_generator_output_is_bounded_to_the_data_range(config):
    """tanh bounds the output to [-1, 1], which is what the data was scaled to."""
    generator, _ = models.build(config)
    out = generator(torch.randn(64, config.latent_dim) * 25)
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_discriminator_returns_one_unbounded_score_per_image(config):
    _, discriminator = models.build(config)
    out = discriminator(torch.randn(7, 1, 28, 28))
    assert out.shape == (7, 1)


def test_discriminator_returns_logits_not_probabilities(config):
    """No sigmoid on the output. Extreme inputs must escape [0, 1].

    This is the property ``BCEWithLogitsLoss`` depends on. If a sigmoid
    were added, the loss would be applied to an already-squashed value and
    the pairing the report describes would be wrong.
    """
    _, discriminator = models.build(config)
    discriminator.eval()
    with torch.no_grad():
        out = discriminator(torch.randn(256, 1, 28, 28) * 50)
    assert out.min() < 0.0 or out.max() > 1.0
    assert not isinstance(discriminator.net[-1], nn.Sigmoid)


def test_no_label_is_ever_an_input(config):
    """Unconditional: the only generator input is the latent vector."""
    generator, discriminator = models.build(config)
    first = generator.net[0]
    assert first.in_features == config.latent_dim

    # And the discriminator takes an image and nothing else.
    z = torch.randn(3, config.latent_dim)
    assert discriminator(generator(z)).shape == (3, 1)


def test_build_is_reproducible_at_a_seed(config):
    """Both runs must start from the same weights."""
    first, _ = models.build(config)
    second, _ = models.build(config)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_contrast_run_starts_from_the_same_weights(config):
    """Changing the discriminator learning rate must not change init."""
    baseline_g, baseline_d = models.build(config)
    contrast_g, contrast_d = models.build(config.contrast(1e-3))
    for a, b in zip(baseline_g.parameters(), contrast_g.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(baseline_d.parameters(), contrast_d.parameters()):
        assert torch.equal(a, b)


def test_parameter_count_is_all_trainable(config):
    generator, discriminator = models.build(config)
    for module in (generator, discriminator):
        counts = models.parameter_count(module)
        assert counts["total"] > 0
        assert counts["trainable"] == counts["total"]
        assert counts["frozen"] == 0
