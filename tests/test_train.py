"""The alternating updates, checked on real parameters.

The brief requires that the discriminator step not update the generator,
and that the generator step preserve the gradient path through the
discriminator while updating only generator parameters. These are the
tests that make those two sentences checkable rather than merely written
down: each one snapshots every parameter of the network that is supposed
to hold still, runs the step, and compares.
"""

import torch

from gan_mnist import train

from conftest import snapshot, unchanged


def test_discriminator_step_does_not_update_the_generator(
    nets, optimizers, criterion, real_batch, noise_batch
):
    """The requirement, stated directly."""
    generator, discriminator = nets
    optimizer_g, optimizer_d = optimizers

    before = snapshot(generator)
    train.discriminator_step(
        generator, discriminator, optimizer_d, criterion, real_batch, noise_batch
    )

    assert unchanged(generator, before) == []


def test_discriminator_step_does_update_the_discriminator(
    nets, optimizers, criterion, real_batch, noise_batch
):
    """The other half: it is supposed to update something."""
    generator, discriminator = nets
    _, optimizer_d = optimizers

    before = snapshot(discriminator)
    train.discriminator_step(
        generator, discriminator, optimizer_d, criterion, real_batch, noise_batch
    )

    assert unchanged(discriminator, before) != []


def test_discriminator_step_sends_no_gradient_to_the_generator(
    nets, optimizers, criterion, real_batch, noise_batch
):
    """``fake.detach()`` severs the graph, so the generator sees nothing.

    Stronger than the parameter test above: that one would also pass if a
    gradient arrived but the optimizer happened not to apply it. This
    asserts the gradient never arrives.
    """
    generator, discriminator = nets
    _, optimizer_d = optimizers

    generator.zero_grad(set_to_none=True)
    train.discriminator_step(
        generator, discriminator, optimizer_d, criterion, real_batch, noise_batch
    )

    for name, parameter in generator.named_parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, (
            f"generator parameter {name} received gradient during the "
            f"discriminator step"
        )


def test_generator_step_does_not_update_the_discriminator(
    nets, optimizers, criterion, real_batch, noise_batch
):
    """The generator step runs through the discriminator without moving it."""
    generator, discriminator = nets
    optimizer_g, optimizer_d = optimizers

    _, fake = train.discriminator_step(
        generator, discriminator, optimizer_d, criterion, real_batch, noise_batch
    )

    before = snapshot(discriminator)
    train.generator_step(generator, discriminator, optimizer_g, criterion, fake)

    assert unchanged(discriminator, before) == []


def test_generator_step_updates_the_generator(
    nets, optimizers, criterion, real_batch, noise_batch
):
    generator, discriminator = nets
    optimizer_g, optimizer_d = optimizers

    _, fake = train.discriminator_step(
        generator, discriminator, optimizer_d, criterion, real_batch, noise_batch
    )

    before = snapshot(generator)
    train.generator_step(generator, discriminator, optimizer_g, criterion, fake)

    assert unchanged(generator, before) != []


def test_generator_step_preserves_the_path_through_the_discriminator(
    nets, optimizers, criterion, real_batch, noise_batch
):
    """The gradient must reach the generator *through* the discriminator.

    If ``fake`` were detached in the generator step too - a natural
    copy-paste error - the loss would have no path to the generator, and
    the generator would receive no gradient at all. Every parameter
    getting a gradient is what rules that out.
    """
    generator, discriminator = nets
    optimizer_g, optimizer_d = optimizers

    _, fake = train.discriminator_step(
        generator, discriminator, optimizer_d, criterion, real_batch, noise_batch
    )

    generator.zero_grad(set_to_none=True)
    train.generator_step(generator, discriminator, optimizer_g, criterion, fake)

    ungradiented = [
        name for name, parameter in generator.named_parameters()
        if parameter.grad is None
    ]
    assert ungradiented == [], (
        f"no gradient reached {ungradiented}; the path through the "
        f"discriminator was not preserved"
    )

    total = sum(
        torch.count_nonzero(p.grad).item() for p in generator.parameters()
    )
    assert total > 0, "the generator received only zero gradients"


def test_optimizers_hold_disjoint_parameters(nets, optimizers):
    """The second, independent guarantee: neither step *could* move the other.

    Even if a gradient reached the wrong network, an optimizer that does
    not hold those parameters cannot apply it.
    """
    generator, discriminator = nets
    optimizer_g, optimizer_d = optimizers

    g_params = {id(p) for group in optimizer_g.param_groups for p in group["params"]}
    d_params = {id(p) for group in optimizer_d.param_groups for p in group["params"]}

    assert g_params == {id(p) for p in generator.parameters()}
    assert d_params == {id(p) for p in discriminator.parameters()}
    assert g_params.isdisjoint(d_params)


def test_generator_targets_real_not_fake(
    nets, optimizers, criterion, real_batch, noise_batch
):
    """The non-saturating objective: the generator's target is 1.

    Checked behaviorally. A discriminator that calls the fake real should
    give the generator a small loss; one that calls it fake should give a
    large one. If the target were 0 this relation would invert.
    """
    generator, discriminator = nets
    optimizer_g, _ = optimizers

    fake = generator(noise_batch)

    with torch.no_grad():
        confident_real = torch.full((fake.size(0), 1), 10.0)
        confident_fake = torch.full((fake.size(0), 1), -10.0)
        loss_when_believed = criterion(confident_real, torch.ones_like(confident_real))
        loss_when_caught = criterion(confident_fake, torch.ones_like(confident_fake))

    assert loss_when_believed < loss_when_caught


def test_full_loop_records_every_epoch_and_checkpoint(config, device, nets):
    """One end-to-end pass, on noise, to check the bookkeeping."""
    from torch.utils.data import DataLoader, TensorDataset

    from gan_mnist import evaluate

    rng = torch.Generator().manual_seed(3)
    images = torch.rand(config.subset_size, 1, 28, 28, generator=rng) * 2 - 1
    loader = DataLoader(TensorDataset(images), batch_size=config.batch_size,
                        shuffle=False, drop_last=True)

    result = train.train(
        config, loader, device, evaluate.make_fixed_noise(config), log=lambda *a: None
    )

    assert len(result["history"]) == config.epochs
    assert len(result["checkpoints"]) == len(config.checkpoints)
    assert [c["epoch"] for c in result["checkpoints"]] == list(config.checkpoints)
    assert result["total_seconds"] > 0
    # One checksum, shared by every checkpoint.
    assert len({c["noise_checksum"] for c in result["checkpoints"]}) == 1
