"""The MNIST subset: how it is chosen, recorded, and rebuilt.

The brief forbids submitting dataset downloads, so the images are not in
the repository. What is committed is ``data/subset_manifest.json`` - the
10,000 indices, the seed that produced them and a checksum over them - and
the subset is rebuilt from that. Both experiments load through here, and
``validate_results.py`` checks they recorded the same checksum, which is
what makes "the same data subset" a verified claim rather than an intention.

Normalization is to [-1, 1] to match the generator's tanh output. A
generator whose outputs are bounded to [-1, 1] cannot reproduce data in
[0, 1]: half its range would be unusable and the discriminator would
separate real from fake on brightness alone.
"""

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from ._config import DATA_DIR, MANIFEST_PATH, RunConfig

# ToTensor gives [0, 1]; this shifts to [-1, 1].
NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def select_indices(seed: int, subset_size: int, population: int) -> np.ndarray:
    """Choose the subset deterministically.

    NumPy's PCG64 rather than torch's generator because this has to give
    the same answer on any device, and torch's RNG streams differ between
    CPU and MPS. The subset must not depend on which machine built it.
    """
    if subset_size > population:
        raise ValueError(
            f"Requested {subset_size} images but MNIST train has {population}"
        )
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(population, size=subset_size, replace=False))


def checksum(indices: np.ndarray) -> str:
    """A checksum over the chosen indices, not over the pixels.

    The pixels are fixed by MNIST; what a run could get wrong is which of
    them it trained on.
    """
    canonical = ",".join(str(int(i)) for i in indices).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def download(root=DATA_DIR):
    """Fetch MNIST if it is not already present. Returns the train split."""
    root.mkdir(parents=True, exist_ok=True)
    return datasets.MNIST(root=str(root), train=True, download=True,
                          transform=NORMALIZE)


def write_manifest(config: RunConfig, path=MANIFEST_PATH) -> dict:
    """Select the subset and record it so it can be rebuilt without the data."""
    train = download()
    indices = select_indices(config.seed, config.subset_size, len(train))
    manifest = {
        "dataset": "MNIST train split",
        "population": len(train),
        "subset_size": int(config.subset_size),
        "seed": int(config.seed),
        "selection": "numpy.random.default_rng(seed).choice, without replacement, sorted",
        "normalization": "ToTensor to [0, 1], then (x - 0.5) / 0.5 to [-1, 1]",
        "image_shape": [1, 28, 28],
        "checksum": checksum(indices),
        "indices": [int(i) for i in indices],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_manifest(path=MANIFEST_PATH) -> dict:
    if not path.is_file():
        raise SystemExit(
            "No subset manifest. Build it first:\n"
            "  .venv/bin/python scripts/build_mnist_subset.py"
        )
    return json.loads(path.read_text())


def load_subset(config: RunConfig) -> tuple:
    """Return (TensorDataset, manifest).

    The whole subset is materialized into one tensor rather than read
    through the torchvision dataset each epoch. 10,000 28x28 floats is
    31 MB, and doing it once removes per-batch decode from the timings
    the report has to explain.
    """
    manifest = read_manifest()
    if manifest["subset_size"] != config.subset_size:
        raise SystemExit(
            f"Manifest holds {manifest['subset_size']} images but the run "
            f"asks for {config.subset_size}. Rebuild it:\n"
            "  .venv/bin/python scripts/build_mnist_subset.py"
        )

    train = download()
    indices = np.array(manifest["indices"], dtype=np.int64)
    if checksum(indices) != manifest["checksum"]:
        raise SystemExit("Manifest checksum does not match its own indices.")

    images = torch.stack([train[int(i)][0] for i in indices])
    return TensorDataset(images), manifest


def make_loader(dataset, config: RunConfig) -> DataLoader:
    """A loader whose shuffling is reproducible and device-independent.

    The generator is seeded on the CPU deliberately: torch's MPS RNG is a
    different stream, and the batch order should not change because the
    run moved to a different device.
    """
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
