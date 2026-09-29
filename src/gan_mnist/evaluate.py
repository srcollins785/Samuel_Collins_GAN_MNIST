"""The fixed evaluation noise, and turning it into images.

The brief asks that the same 16 noise vectors be used at every checkpoint,
so that what changes between grids is the generator and nothing else. That
only means something if it is verifiable, so the vectors are hashed and the
hash is written into every checkpoint record; ``validate_results.py``
fails unless all of them - across both runs - are identical.

The noise is generated on the CPU and moved to the device afterwards.
torch's MPS generator is a different stream from its CPU generator, so
seeding alone would not give the same vectors on a machine that resolved
to a different device.
"""

import hashlib

import torch


def make_fixed_noise(config) -> torch.Tensor:
    """The evaluation vectors. Same seed, same device, same numbers, always.

    Seeded from ``seed + 1`` rather than ``seed`` so the evaluation noise
    is not the opening slice of the training noise stream. Nothing depends
    on that, but a grid drawn from vectors the generator was also trained
    against that step would be a slightly flattering sample.
    """
    generator = torch.Generator()
    generator.manual_seed(config.seed + 1)
    return torch.randn(
        config.n_eval_samples, config.latent_dim, generator=generator
    )


def noise_checksum(noise: torch.Tensor) -> str:
    """A checksum over the evaluation vectors themselves."""
    contiguous = noise.detach().to("cpu").contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def training_noise(batch_size: int, latent_dim: int, generator) -> torch.Tensor:
    """A batch of training noise from an explicit CPU generator.

    Passing the generator rather than relying on the global RNG keeps the
    noise stream independent of anything else that draws random numbers,
    so the two runs see the same sequence of latent vectors.
    """
    return torch.randn(batch_size, latent_dim, generator=generator)


@torch.no_grad()
def generate(generator_net, noise: torch.Tensor, device) -> torch.Tensor:
    """Images from noise, in eval mode, detached, back on the CPU.

    Eval mode matters more than it looks: the generator has BatchNorm, and
    in training mode a 16-sample grid would be normalized by the statistics
    of those 16 samples. The grid would then depend on which samples share
    it, and checkpoints would not be comparable.
    """
    was_training = generator_net.training
    generator_net.eval()
    images = generator_net(noise.to(device)).detach().to("cpu")
    generator_net.train(was_training)
    return images


def discriminator_accuracy(real_logits, fake_logits) -> dict:
    """How often the discriminator is right, at a threshold of zero.

    A logit above zero is a probability above 0.5. Recorded so the report
    can show what a high discriminator accuracy does and does not imply
    about the generator - which is one of the questions the brief asks.
    """
    real_correct = (real_logits > 0).float().mean().item()
    fake_correct = (fake_logits <= 0).float().mean().item()
    return {
        "real": real_correct,
        "fake": fake_correct,
        "overall": (real_correct + fake_correct) / 2,
    }
