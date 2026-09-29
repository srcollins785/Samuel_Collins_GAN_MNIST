"""Figures: the real sample grid, the checkpoint grids, the loss curves.

The brief asks that generated images be clearly labeled, so every grid
carries the run it came from and the epoch it was rendered at, and the
epoch-0 grid says "before training" rather than leaving the reader to
infer what epoch 0 means.

Images arrive in [-1, 1] from the generator's tanh and are shifted back to
[0, 1] for display only. Nothing is rescaled per-image: a grid whose
contrast was stretched sample by sample would hide exactly the washed-out
early output the report has to discuss.
"""

import matplotlib

matplotlib.use("Agg")  # no display; these are written to files

import matplotlib.pyplot as plt
import torch

GRID_ROWS = 4
GRID_COLS = 4


def to_display(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] to [0, 1], clamped. Not per-image normalized - see module docstring."""
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


def save_grid(images: torch.Tensor, path, title: str) -> str:
    """A 4x4 grid of 28x28 images, labeled."""
    display = to_display(images)
    figure, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(4.2, 4.5))
    for index, axis in enumerate(axes.flat):
        axis.imshow(display[index].squeeze(0).numpy(), cmap="gray",
                    vmin=0.0, vmax=1.0)
        axis.axis("off")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def checkpoint_title(run_name: str, epoch: int) -> str:
    when = "before training" if epoch == 0 else f"after epoch {epoch}"
    return f"Generated - {run_name} run, {when}"


def save_real_grid(dataset, path, count: int = 16, seed: int = 42) -> str:
    """Sixteen real examples, which the brief asks to be displayed.

    Drawn with an explicit generator so the same sixteen appear every time
    the figure is rebuilt.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng)[:count]
    images = torch.stack([dataset[int(i)][0] for i in indices])
    return save_grid(images, path, "Real MNIST examples (training subset)")


def save_loss_curves(history, path, title: str) -> str:
    """Average generator and discriminator loss per epoch, for one run."""
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    axis.plot(epochs, [r["generator_loss"] for r in history],
              marker="o", label="Generator")
    axis.plot(epochs, [r["discriminator_loss"] for r in history],
              marker="s", label="Discriminator")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Average loss")
    axis.set_title(title, fontsize=10)
    axis.set_xticks(epochs)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def save_comparison_curves(histories: dict, path, title: str) -> str:
    """Both runs on shared axes.

    Generator and discriminator are given separate panels: they are not on
    the same scale, and overlaying four series on one axis makes the one
    difference the experiment is about harder to see rather than easier.
    """
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)
    styles = {"baseline": ("o", "-"), "contrast": ("s", "--")}

    for run_name, history in histories.items():
        marker, line = styles.get(run_name, ("^", ":"))
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [r["generator_loss"] for r in history],
                     marker=marker, linestyle=line, label=run_name)
        axes[1].plot(epochs, [r["discriminator_loss"] for r in history],
                     marker=marker, linestyle=line, label=run_name)
        axes[0].set_xticks(epochs)

    axes[0].set_title("Generator loss", fontsize=10)
    axes[1].set_title("Discriminator loss", fontsize=10)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Average loss")
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def save_accuracy_curves(histories: dict, path, title: str) -> str:
    """Discriminator accuracy per epoch, both runs.

    Included because the report has to argue that a high discriminator
    accuracy does not by itself establish a bad generator, or a low one a
    good generator, and that argument is easier to make against a plotted
    number than an asserted one.
    """
    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    styles = {"baseline": ("o", "-"), "contrast": ("s", "--")}
    for run_name, history in histories.items():
        marker, line = styles.get(run_name, ("^", ":"))
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [r["discriminator_accuracy"] for r in history],
                  marker=marker, linestyle=line, label=run_name)
        axis.set_xticks(epochs)
    axis.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    axis.annotate("chance", xy=(epochs[0], 0.5), xytext=(0, 4),
                  textcoords="offset points", fontsize=8, color="gray")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Discriminator accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)
