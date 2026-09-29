"""Subset selection, normalization and the manifest.

The tests that need the MNIST download skip when it is absent, so the
suite runs on a clean checkout. The tests that matter most - that the
selection is deterministic and that the checksum detects a changed
subset - need no data at all.
"""

import numpy as np
import pytest
import torch

from gan_mnist import data
from gan_mnist._config import MANIFEST_PATH, RunConfig

needs_mnist = pytest.mark.skipif(
    not MANIFEST_PATH.is_file(),
    reason="run scripts/build_mnist_subset.py first",
)


def test_selection_is_deterministic():
    first = data.select_indices(42, 1000, 60000)
    second = data.select_indices(42, 1000, 60000)
    assert np.array_equal(first, second)


def test_selection_is_device_independent_by_construction():
    """NumPy, not torch: torch's CPU and MPS streams differ."""
    indices = data.select_indices(42, 100, 60000)
    assert indices.dtype.kind == "i"
    assert len(set(indices.tolist())) == 100


def test_a_different_seed_selects_a_different_subset():
    assert not np.array_equal(
        data.select_indices(42, 1000, 60000),
        data.select_indices(43, 1000, 60000),
    )


def test_selection_is_sorted_and_without_replacement():
    indices = data.select_indices(42, 5000, 60000)
    assert list(indices) == sorted(indices)
    assert len(set(indices.tolist())) == 5000


def test_oversized_subset_is_refused():
    with pytest.raises(ValueError):
        data.select_indices(42, 70000, 60000)


def test_checksum_detects_a_changed_subset():
    original = data.select_indices(42, 1000, 60000)
    altered = original.copy()
    altered[0] = altered[0] + 1
    assert data.checksum(original) != data.checksum(altered)


def test_checksum_is_stable_across_calls():
    indices = data.select_indices(42, 1000, 60000)
    assert data.checksum(indices) == data.checksum(indices.copy())


@needs_mnist
def test_manifest_matches_its_own_indices():
    manifest = data.read_manifest()
    indices = np.array(manifest["indices"], dtype=np.int64)
    assert data.checksum(indices) == manifest["checksum"]
    assert len(indices) == manifest["subset_size"]


@needs_mnist
def test_manifest_is_reproducible_from_its_recorded_seed():
    manifest = data.read_manifest()
    rebuilt = data.select_indices(
        manifest["seed"], manifest["subset_size"], manifest["population"]
    )
    assert data.checksum(rebuilt) == manifest["checksum"]


@needs_mnist
def test_loaded_images_are_normalized_to_the_generator_range():
    """[-1, 1], matching tanh. Anything else and half the range is dead."""
    dataset, _ = data.load_subset(RunConfig())
    images = torch.stack([dataset[i][0] for i in range(256)])
    assert images.min() >= -1.0
    assert images.max() <= 1.0
    # MNIST has saturated black and white pixels, so both ends are reached.
    assert images.min() < -0.95
    assert images.max() > 0.95


@needs_mnist
def test_loaded_subset_has_the_requested_size_and_shape():
    config = RunConfig()
    dataset, manifest = data.load_subset(config)
    assert len(dataset) == config.subset_size
    assert dataset[0][0].shape == (1, 28, 28)
    assert manifest["checksum"] == data.read_manifest()["checksum"]


@needs_mnist
def test_loader_batches_are_reproducible():
    """Both runs must see the same batches in the same order."""
    config = RunConfig(subset_size=10_000, batch_size=64)
    dataset, _ = data.load_subset(config)
    first = next(iter(data.make_loader(dataset, config)))[0]
    second = next(iter(data.make_loader(dataset, config)))[0]
    assert torch.equal(first, second)


@needs_mnist
def test_loader_drops_the_partial_final_batch():
    config = RunConfig(subset_size=10_000, batch_size=128)
    dataset, _ = data.load_subset(config)
    loader = data.make_loader(dataset, config)
    assert len(loader) == config.batches_per_epoch
    for (batch,) in loader:
        assert batch.size(0) == config.batch_size
