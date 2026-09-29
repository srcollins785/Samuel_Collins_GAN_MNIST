# Samuel_Collins_GAN_MNIST

A small unconditional generative adversarial network trained on a reproducible
10,000-image subset of MNIST, with one controlled comparison: the same
experiment repeated from scratch at a different discriminator learning rate,
every other setting held fixed.

Lab Assignment 5, Computer Vision. Clark Atlanta University, Department of
Cyber-Physical Systems. Instructor: Dr. Kishor Datta Gupta.

**Result.** The run with the *lower* generator loss produced the *worse*
images. Lowering the discriminator learning rate from 2e-4 to 2e-5 left the
discriminator near chance, and a generator optimizing against an uninformative
critic reached a final loss of 0.6365 against the baseline's 0.9104 while
producing speckled clouds instead of digit strokes. Generator loss and sample
quality moved in opposite directions.

- Report: `report/GAN_MNIST_Report.pdf`
- Executed notebook: `GAN_MNIST.ipynb`
- Machine-written configuration: `configuration.yaml`

## What this repository is arranged to prove

Two claims carry most of the assignment's weight, and both are checked
mechanically rather than asserted in prose:

- **The alternating updates are correct.** `tests/test_train.py` snapshots
  every parameter of each network, runs one update step of the other, and
  asserts the snapshot is unchanged - so "the discriminator step does not
  update the generator" is an executable assertion, not a comment. Removing
  the `.detach()` fails five of those tests, which was verified deliberately:
  a test that cannot fail establishes nothing.
- **The comparison is controlled.** `scripts/validate_results.py` diffs the
  two saved run configurations and fails unless the discriminator learning
  rate is the *only* field that differs, and unless the 16 fixed evaluation
  noise vectors hash identically across every checkpoint of both runs.

## Installation

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

No GPU is required. The device is selected automatically and recorded in the
run configuration. MPS was measured at 2.1 s against 2.8 s for CPU over five
epochs on an Apple M4 Max.

## Running everything

    .venv/bin/python scripts/run_all.py

Tests, subset, both experiments, the supplementary runs, the audit, the
report, the PDF, and the executed notebook. About 100 seconds end to end.
The first failure stops the run and nothing after it executes.

    .venv/bin/python scripts/run_all.py --skip-training

Rebuilds the documents from existing results without retraining, which is
what most edits need.

### Individual steps

    .venv/bin/python scripts/build_mnist_subset.py    # download and record the subset
    .venv/bin/python run_experiment.py --run both     # the two graded runs
    .venv/bin/python scripts/run_supplementary.py     # sweep and seed replication
    .venv/bin/python scripts/validate_results.py      # audit the saved results
    .venv/bin/python scripts/generate_report.py       # Markdown, from results/
    .venv/bin/python scripts/build_report_pdf.py      # PDF, via headless Chrome
    .venv/bin/python scripts/build_notebook.py --execute

## The data

The images are not in the repository - the brief forbids submitting dataset
downloads. What is committed is `data/subset_manifest.json`: the 10,000
indices, the seed that chose them, and a checksum over them.
`scripts/build_mnist_subset.py` downloads MNIST and rebuilds the same subset
from the same seed, and the audit checks both runs recorded the same checksum.

Selection uses NumPy and the evaluation noise is drawn on the CPU before being
moved to the device, because torch's MPS and CPU generators are different
streams. Neither the subset nor the evaluation grid should change because the
run moved to another machine. Losses reproduce bit-for-bit across reruns.

## Reading the results

`report/GAN_MNIST_Report.pdf` is generated from `results/*.json` by
`scripts/generate_report.py`, which imports nothing from the package. The
experiments communicate with the report through files, so the document can be
rebuilt in a second without retraining and cannot describe a result that was
not produced. Every number in it, including the headline stating which run
won, is a query into `scripts/report_data.py` rather than a sentence someone
remembered.

The analysis section is written prose - judgments about what the images look
like are visual readings of sixteen-image grids and are labeled as such - but
every number inside it is a query, and the audit fails the build if the
section falls outside the brief's 400 to 600 words.

## Documented choices

- **Batch size 32, not 128.** The brief fixes 10,000 images and 5 epochs but
  not the batch size, and that choice sets how many times the generator is
  actually updated: 390 at batch 128 against 1,560 at batch 32. At 128 the
  five-epoch samples are centered blobs; at 32 they are recognizable strokes.
  Batch 16 was no better and twice as slow.
- **The standard compute path.** The reduced 5,000-image, 3-epoch option was
  not needed; the full run takes about seven seconds.
- **Four discriminator rates were scouted** before the contrast value was
  chosen, and all four are reported in the report's Section 6 rather than only
  the one selected. Two of them moved the opposite way.
- **Sample quality is judged visually.** No FID or classifier-based score was
  computed, and the report says so wherever it calls an image better or worse.

## Layout

    src/gan_mnist/     the package: config, data, models, training, evaluation, plots
    scripts/           subset build, supplementary runs, audit, report, notebook, orchestration
    tests/             pytest suite (47 tests)
    run_experiment.py  single entry point for the two graded experiments
    GAN_MNIST.ipynb    executed notebook, a thin wrapper over src/
    results/           what the runs recorded; the report reads only this
    plots/             figures, byte-identical across reruns
    report/            generated Markdown and PDF
    configuration.yaml generated from results/, not written by hand

`results/`, `plots/` and `report/` are gitignored by default and force-added
once the real runs complete. A report built from a scouting run reads exactly
like one built from the real thing, so a number entering the repository is a
deliberate act.

## Tests

    .venv/bin/python -m pytest

47 tests: parameter isolation across the two update steps, the logits/loss
pairing, subset determinism and checksum sensitivity, fixed-noise stability
across runs, and table formatting.

## Source

Adapted from the official TensorFlow DCGAN tutorial
(<https://www.tensorflow.org/tutorials/generative/dcgan>). Changes and reasons
are listed in Section 8 of the report.

## License

MIT. See `LICENSE`.
