"""Run configuration and the paths everything else reads and writes.

One frozen dataclass per experiment. The two experiments this assignment
requires differ in exactly one field, and ``COMPARED_FIELDS`` names the
fields that must be identical between them - ``scripts/validate_results.py``
diffs the saved configurations against that list and fails if anything but
the discriminator learning rate moved. Keeping the list here, beside the
definition, means adding a knob without deciding whether it is part of the
controlled comparison is not possible.
"""

from dataclasses import dataclass, asdict, replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"
REPORT_DIR = REPO_ROOT / "report"
MANIFEST_PATH = DATA_DIR / "subset_manifest.json"

# Every field that the controlled comparison holds fixed. The discriminator
# learning rate is deliberately absent: it is the one thing allowed to vary.
# ``name``, ``device`` and ``epochs`` are handled separately - name differs
# by construction, device is a property of the machine rather than of the
# experiment, and epochs is checked against the recorded history length.
COMPARED_FIELDS = (
    "seed",
    "subset_size",
    "batch_size",
    "latent_dim",
    "epochs",
    "generator_lr",
    "beta1",
    "beta2",
    "checkpoints",
    "n_eval_samples",
)


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines a run, and nothing that does not.

    Defaults are the baseline. The contrast run is produced by
    ``baseline.contrast(discriminator_lr=...)``, which changes that one
    field and the name, so a second experiment cannot accidentally differ
    in a second way.
    """

    name: str = "baseline"
    seed: int = 42
    subset_size: int = 10_000
    # 32 rather than a more usual 128. The brief fixes the budget at five
    # epochs over 10,000 images, which is 390 generator updates at batch 128
    # and 1,560 at batch 32. Batch size is not one of the things the brief
    # fixes, and a scouting sweep (recorded in the report) found batch 32
    # produced recognizable digit strokes at five epochs where batch 128
    # produced centered blobs. Batch 16 was no better and twice as slow.
    batch_size: int = 32
    latent_dim: int = 100
    epochs: int = 5
    generator_lr: float = 2e-4
    # Equal to the generator rate, which is what the TensorFlow DCGAN
    # tutorial this work adapts uses. The contrast run moves this and only
    # this.
    discriminator_lr: float = 2e-4
    # Adam at beta1=0.5 rather than the 0.9 default. The DCGAN paper found
    # 0.9 made training oscillate; 0.5 is the value the TensorFlow tutorial
    # this work adapts also uses.
    beta1: float = 0.5
    beta2: float = 0.999
    # Epoch numbers at which the fixed-noise grid is rendered. 0 means
    # before any training has happened, which the brief asks for.
    checkpoints: tuple = (0, 1, 3, 5)
    n_eval_samples: int = 16
    device: str = "auto"

    def contrast(self, discriminator_lr: float, name: str = "contrast"):
        """The controlled comparison: this run with one field moved."""
        return replace(self, name=name, discriminator_lr=discriminator_lr)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["checkpoints"] = list(data["checkpoints"])
        return data

    @property
    def batches_per_epoch(self) -> int:
        # drop_last=True in the loader, so a partial final batch is not run.
        return self.subset_size // self.batch_size


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick the device, preferring Metal when it is available.

    ``auto`` is not automatically the fastest choice for a model this
    small - see the timing recorded in ``configuration.yaml`` - so the
    resolved device is always written into the run's saved configuration
    rather than inferred later from the machine.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
