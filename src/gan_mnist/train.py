"""The alternating updates.

This is the part of the assignment with the most weight on it, and the
requirement is specific: prevent generator updates during the
discriminator step, and during the generator step preserve the gradient
path through the discriminator to the generator while updating only
generator parameters. The two steps are separate functions rather than
one loop body so that ``tests/test_train.py`` can call each in isolation
and assert, on real parameters, that the other network did not move.

Two independent mechanisms keep each step honest, and both are deliberate:

1. ``fake.detach()`` in the discriminator step severs the graph, so no
   gradient reaches the generator at all.
2. Each optimizer is constructed over one network's parameters only, so
   even a gradient that did arrive could not be applied by the wrong step.

The generator step relies on the opposite of (1): it feeds the
*non-detached* ``fake`` to the discriminator, so the chain rule runs
discriminator-then-generator and the generator learns from how the
discriminator scored it.
"""

import time

import torch
from torch import nn

from . import evaluate, models


def _labels(logits: torch.Tensor, value: float) -> torch.Tensor:
    """Targets shaped like the logits they are scored against."""
    return torch.full_like(logits, value)


def discriminator_step(generator, discriminator, optimizer, criterion,
                       real, noise):
    """Update the discriminator. Returns (metrics, fake).

    ``fake`` is returned still carrying its graph back to the generator,
    because the generator step needs that path. What the discriminator
    step sees is ``fake.detach()``.

    The ``zero_grad`` here also clears the discriminator gradients that
    the *previous* generator step deposited on the way through. Those are
    never applied - the generator optimizer cannot see discriminator
    parameters - but leaving them to accumulate into the next real
    discriminator update would be a genuine bug.
    """
    optimizer.zero_grad(set_to_none=True)

    real_logits = discriminator(real)
    loss_real = criterion(real_logits, _labels(real_logits, 1.0))

    fake = generator(noise)
    # detach: the discriminator is told to score this image, not to teach
    # the generator how to have made it.
    fake_logits = discriminator(fake.detach())
    loss_fake = criterion(fake_logits, _labels(fake_logits, 0.0))

    loss = loss_real + loss_fake
    loss.backward()
    # optimizer holds discriminator parameters only.
    optimizer.step()

    accuracy = evaluate.discriminator_accuracy(real_logits, fake_logits)
    return {"loss": loss.item(), "accuracy": accuracy}, fake


def generator_step(generator, discriminator, optimizer, criterion, fake):
    """Update the generator. Returns metrics.

    The discriminator is run on the *non-detached* fake, so backward
    travels through the discriminator into the generator. The
    discriminator accumulates gradients during that traversal and they are
    simply never applied: this optimizer holds generator parameters only,
    and the next discriminator step zeroes them before it starts.

    The target is 1, not 0. The generator is not minimizing the
    discriminator's success; it is maximizing the probability that its own
    output is called real. This is the non-saturating form, which gives a
    strong gradient exactly when the generator is losing badly - the
    moment it most needs one.
    """
    optimizer.zero_grad(set_to_none=True)

    logits = discriminator(fake)
    loss = criterion(logits, _labels(logits, 1.0))
    loss.backward()
    # optimizer holds generator parameters only.
    optimizer.step()

    return {"loss": loss.item()}


def train(config, loader, device, fixed_noise, on_checkpoint=None,
          log=print) -> dict:
    """Run one experiment end to end and return everything it measured.

    ``on_checkpoint(epoch, images)`` is called with the fixed-noise grid at
    epoch 0 - before any training - and after each epoch named in
    ``config.checkpoints``. Rendering is the caller's job; this function
    stays responsible only for producing the images at the right moments.
    """
    generator, discriminator = models.build(config)
    generator.to(device)
    discriminator.to(device)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=config.generator_lr, betas=(config.beta1, config.beta2),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.discriminator_lr, betas=(config.beta1, config.beta2),
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training noise from its own CPU generator, so the sequence of latent
    # vectors is identical in both runs regardless of device.
    noise_rng = torch.Generator()
    noise_rng.manual_seed(config.seed + 2)

    checksum = evaluate.noise_checksum(fixed_noise)
    history = []
    checkpoints = []

    def checkpoint(epoch):
        images = evaluate.generate(generator, fixed_noise, device)
        record = {
            "epoch": epoch,
            "noise_checksum": checksum,
            "image_min": float(images.min()),
            "image_max": float(images.max()),
        }
        if on_checkpoint is not None:
            record.update(on_checkpoint(epoch, images) or {})
        checkpoints.append(record)

    if 0 in config.checkpoints:
        checkpoint(0)

    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        generator.train()
        discriminator.train()

        totals = {"g": 0.0, "d": 0.0, "real": 0.0, "fake": 0.0, "overall": 0.0}
        batches = 0

        for (real,) in loader:
            real = real.to(device)
            noise = evaluate.training_noise(
                real.size(0), config.latent_dim, noise_rng
            ).to(device)

            d_metrics, fake = discriminator_step(
                generator, discriminator, optimizer_d, criterion, real, noise
            )
            g_metrics = generator_step(
                generator, discriminator, optimizer_g, criterion, fake
            )

            totals["d"] += d_metrics["loss"]
            totals["g"] += g_metrics["loss"]
            totals["real"] += d_metrics["accuracy"]["real"]
            totals["fake"] += d_metrics["accuracy"]["fake"]
            totals["overall"] += d_metrics["accuracy"]["overall"]
            batches += 1

        seconds = time.perf_counter() - epoch_started
        history.append({
            "epoch": epoch,
            "generator_loss": totals["g"] / batches,
            "discriminator_loss": totals["d"] / batches,
            "discriminator_accuracy_real": totals["real"] / batches,
            "discriminator_accuracy_fake": totals["fake"] / batches,
            "discriminator_accuracy": totals["overall"] / batches,
            "batches": batches,
            "seconds": seconds,
        })
        log(
            f"  epoch {epoch}/{config.epochs}  "
            f"G {history[-1]['generator_loss']:.4f}  "
            f"D {history[-1]['discriminator_loss']:.4f}  "
            f"D-acc {history[-1]['discriminator_accuracy']:.3f}  "
            f"{seconds:.1f}s"
        )

        if epoch in config.checkpoints:
            checkpoint(epoch)

    total_seconds = time.perf_counter() - started

    return {
        "config": config.to_dict(),
        "device": str(device),
        "fixed_noise_checksum": checksum,
        "history": history,
        "checkpoints": checkpoints,
        "total_seconds": total_seconds,
        "parameters": {
            "generator": models.parameter_count(generator),
            "discriminator": models.parameter_count(discriminator),
        },
    }
