# Samuel_Collins_GAN_MNIST

A small unconditional generative adversarial network trained on a reproducible
10,000-image subset of MNIST, with one controlled comparison: the same
experiment repeated from scratch at a different discriminator learning rate,
every other setting held fixed.

Lab Assignment 5, Computer Vision. Clark Atlanta University, Department of
Cyber-Physical Systems. Instructor: Dr. Kishor Datta Gupta.

The generator maps a 100-dimensional noise vector to a 28x28 image; the
discriminator scores an image as real or generated. Digit labels are never an
input. Both networks are small fully connected stacks, which the brief permits
and which trains in under two minutes on the hardware recorded in
`configuration.yaml`.

## What this repository is arranged to prove

Two claims carry most of the assignment's weight, and both are checked
mechanically rather than asserted in prose:

- **The alternating updates are correct.** `tests/test_train.py` snapshots
  every parameter of each network, runs one update step of the other, and
  asserts the snapshot is unchanged - so "the discriminator step does not
  update the generator" is an executable assertion, not a comment.
- **The comparison is controlled.** `scripts/validate_results.py` diffs the
  two saved run configurations and fails unless the discriminator learning
  rate is the *only* field that differs, and unless the 16 fixed evaluation
  noise vectors hash identically across every checkpoint of both runs.

## Installation

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

No GPU is required. The device is selected automatically and recorded in the
run configuration.

## Building the dataset

    .venv/bin/python scripts/build_mnist_subset.py

Downloads MNIST into `data/` (gitignored - the brief forbids submitting
dataset downloads) and selects the 10,000-image subset at seed 42, writing
`data/subset_manifest.json`. The manifest holds the indices and a checksum, so
the subset is rebuildable byte-for-byte without the images being in the
repository.

## Running the experiments

    .venv/bin/python run_experiment.py --run both

Trains the baseline and the contrast run from scratch, writing per-epoch
losses, checkpoint grids, and timings into `results/` and `plots/`.

The whole pipeline, in the order it must run:

    .venv/bin/python scripts/run_all.py

## Reading the results

    report/GAN_MNIST_Report.pdf

## Layout

    src/gan_mnist/     the package: data, models, training, evaluation, plots
    scripts/           dataset build, validation, report generation, orchestration
    tests/             pytest suite
    run_experiment.py  single entry point for the two experiments
    GAN_MNIST.ipynb    executed notebook (thin: imports from src/)

## License

MIT. See `LICENSE`.
