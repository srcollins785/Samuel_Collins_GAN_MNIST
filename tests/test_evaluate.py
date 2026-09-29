"""The fixed evaluation noise, and what a checkpoint grid is allowed to depend on."""

import torch

from gan_mnist import evaluate, models


def test_fixed_noise_is_identical_every_time(config):
    first = evaluate.make_fixed_noise(config)
    second = evaluate.make_fixed_noise(config)
    assert torch.equal(first, second)
    assert first.shape == (config.n_eval_samples, config.latent_dim)


def test_fixed_noise_is_identical_across_the_two_runs(config):
    """The comparison holds the evaluation noise fixed, so this must hold."""
    baseline = evaluate.make_fixed_noise(config)
    contrast = evaluate.make_fixed_noise(config.contrast(1e-3))
    assert torch.equal(baseline, contrast)
    assert evaluate.noise_checksum(baseline) == evaluate.noise_checksum(contrast)


def test_checksum_changes_when_the_noise_does(config):
    """A checksum that never changes would prove nothing."""
    noise = evaluate.make_fixed_noise(config)
    disturbed = noise.clone()
    disturbed[0, 0] += 1e-3
    assert evaluate.noise_checksum(noise) != evaluate.noise_checksum(disturbed)


def test_generate_leaves_the_generator_in_training_mode(config):
    """Rendering a checkpoint must not silently disable BatchNorm updates."""
    generator, _ = models.build(config)
    generator.train()
    evaluate.generate(generator, evaluate.make_fixed_noise(config),
                      torch.device("cpu"))
    assert generator.training


def test_generate_is_deterministic_for_a_frozen_generator(config):
    """Same weights, same noise, same image - grids are comparable."""
    generator, _ = models.build(config)
    noise = evaluate.make_fixed_noise(config)
    device = torch.device("cpu")
    first = evaluate.generate(generator, noise, device)
    second = evaluate.generate(generator, noise, device)
    assert torch.equal(first, second)


def test_discriminator_accuracy_thresholds_at_zero(config):
    """A logit above zero is a probability above one half."""
    real = torch.tensor([[1.0], [2.0], [-1.0], [0.5]])
    fake = torch.tensor([[-1.0], [-2.0], [3.0], [-0.5]])
    accuracy = evaluate.discriminator_accuracy(real, fake)
    assert accuracy["real"] == 0.75
    assert accuracy["fake"] == 0.75
    assert accuracy["overall"] == 0.75
