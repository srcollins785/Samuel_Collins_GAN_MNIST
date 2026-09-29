"""Audit the published results for internal consistency.

Carried over from the Assignment 2 benchmark, where the equivalent script
was written after two errors survived a round of checking that only
asserted fields were present and positive. A field can be present, and
plausible, and still impossible next to the field beside it. So the checks
here are identities between two recorded quantities rather than existence
checks, and the useful ones are the identities that are cheap to state and
hard to satisfy by accident:

  noise checksum identical       across every checkpoint of both runs
  configurations differ in one   field, and it is the discriminator rate
  history length == epochs       the run ran as long as it claims
  sum(epoch seconds) ~= total    the clock and the epochs agree
  batches == subset // batch     the loader saw the subset it was given
  subset checksum identical      both runs trained on the same images
  generator output in [-1, 1]    tanh bounds it, so a value outside is a bug
  every referenced figure exists the report cannot cite a missing file
  analysis word count in range   the brief asks for 400 to 600

The second is the one the assignment rests on. Fifteen of the ninety
points are for a comparison in which one thing changed, and this is what
makes that claim checkable rather than a sentence in the report.

Usage
-----
    .venv/bin/python scripts/validate_results.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gan_mnist._config import COMPARED_FIELDS, RESULTS_DIR  # noqa: E402

REPORT_MD = REPO_ROOT / "report" / "GAN_MNIST_Report.md"
RUNS = ("baseline", "contrast")
TOLERANCE_SECONDS = 0.5
WORD_RANGE = (400, 600)


class Audit:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)

    def report(self) -> int:
        print(f"\n{self.checks} checks run")
        if not self.failures:
            print("All passed.")
            return 0
        print(f"{len(self.failures)} FAILED:\n")
        for failure in self.failures:
            print(f"  - {failure}")
        return 1


def load_runs() -> dict:
    results = {}
    for name in RUNS:
        path = RESULTS_DIR / f"{name}.json"
        if not path.is_file():
            raise SystemExit(
                f"Missing {path.relative_to(REPO_ROOT)}. Run the experiments "
                f"first:\n  .venv/bin/python run_experiment.py --run both"
            )
        results[name] = json.loads(path.read_text())
    return results


def audit_noise(audit: Audit, results: dict):
    """The same sixteen vectors at every checkpoint, in both runs."""
    checksums = set()
    for name, result in results.items():
        run_checksums = {c["noise_checksum"] for c in result["checkpoints"]}
        audit.check(
            len(run_checksums) == 1,
            f"{name}: checkpoints used {len(run_checksums)} different noise "
            f"vectors; the brief requires one fixed set",
        )
        checksums |= run_checksums
        audit.check(
            result["fixed_noise_checksum"] in run_checksums,
            f"{name}: the recorded fixed-noise checksum is not the one the "
            f"checkpoints were rendered from",
        )

    audit.check(
        len(checksums) == 1,
        f"the two runs used different evaluation noise ({len(checksums)} "
        f"distinct checksums); the grids are not comparable",
    )


def audit_controlled_comparison(audit: Audit, results: dict):
    """One field moved. This is the check the comparison depends on."""
    baseline = results["baseline"]["config"]
    contrast = results["contrast"]["config"]

    for field in COMPARED_FIELDS:
        audit.check(
            baseline[field] == contrast[field],
            f"the comparison is not controlled: {field} is "
            f"{baseline[field]!r} in baseline and {contrast[field]!r} in "
            f"contrast, but only the discriminator learning rate may differ",
        )

    audit.check(
        baseline["discriminator_lr"] != contrast["discriminator_lr"],
        "both runs used the same discriminator learning rate; there is no "
        "comparison",
    )
    audit.check(
        results["baseline"]["subset_checksum"] == results["contrast"]["subset_checksum"],
        "the runs trained on different image subsets",
    )


def audit_history(audit: Audit, results: dict):
    for name, result in results.items():
        config = result["config"]
        history = result["history"]

        audit.check(
            len(history) == config["epochs"],
            f"{name}: {len(history)} epochs recorded but the configuration "
            f"says {config['epochs']}",
        )
        audit.check(
            [row["epoch"] for row in history] == list(range(1, len(history) + 1)),
            f"{name}: epoch numbers are not consecutive from 1",
        )

        expected_batches = config["subset_size"] // config["batch_size"]
        for row in history:
            audit.check(
                row["batches"] == expected_batches,
                f"{name} epoch {row['epoch']}: {row['batches']} batches but "
                f"{config['subset_size']} // {config['batch_size']} is "
                f"{expected_batches}",
            )

        clocked = sum(row["seconds"] for row in history)
        audit.check(
            clocked <= result["total_seconds"] + TOLERANCE_SECONDS,
            f"{name}: epochs sum to {clocked:.2f}s but the run reports "
            f"{result['total_seconds']:.2f}s total",
        )
        audit.check(
            result["total_seconds"] - clocked < 5.0,
            f"{name}: {result['total_seconds'] - clocked:.2f}s unaccounted "
            f"for outside the epochs",
        )

        for row in history:
            for field in ("generator_loss", "discriminator_loss"):
                audit.check(
                    row[field] > 0,
                    f"{name} epoch {row['epoch']}: {field} is {row[field]}, "
                    f"which cross-entropy cannot be",
                )
            accuracy = row["discriminator_accuracy"]
            audit.check(
                0.0 <= accuracy <= 1.0,
                f"{name} epoch {row['epoch']}: discriminator accuracy "
                f"{accuracy} is outside [0, 1]",
            )
            midpoint = (row["discriminator_accuracy_real"]
                        + row["discriminator_accuracy_fake"]) / 2
            audit.check(
                abs(midpoint - accuracy) < 1e-9,
                f"{name} epoch {row['epoch']}: overall accuracy {accuracy} is "
                f"not the mean of the real and fake accuracies ({midpoint})",
            )


def audit_checkpoints(audit: Audit, results: dict):
    for name, result in results.items():
        config = result["config"]
        checkpoints = result["checkpoints"]

        audit.check(
            len(checkpoints) == len(config["checkpoints"]),
            f"{name}: {len(checkpoints)} grids for "
            f"{len(config['checkpoints'])} configured checkpoints",
        )
        audit.check(
            [c["epoch"] for c in checkpoints] == list(config["checkpoints"]),
            f"{name}: checkpoint epochs {[c['epoch'] for c in checkpoints]} "
            f"do not match the configured {config['checkpoints']}",
        )
        audit.check(
            0 in [c["epoch"] for c in checkpoints],
            f"{name}: no grid before training, which the brief requires",
        )

        for record in checkpoints:
            audit.check(
                -1.0 <= record["image_min"] and record["image_max"] <= 1.0,
                f"{name} epoch {record['epoch']}: generated pixels in "
                f"[{record['image_min']:.3f}, {record['image_max']:.3f}] "
                f"escape the tanh range [-1, 1]",
            )
            grid = record.get("grid")
            audit.check(grid is not None, f"{name} epoch {record['epoch']}: "
                                          f"no figure recorded")
            if grid:
                path = REPO_ROOT / grid
                audit.check(
                    path.is_file() and path.stat().st_size > 0,
                    f"{name} epoch {record['epoch']}: {grid} is missing or empty",
                )


def audit_report(audit: Audit):
    """Only meaningful once the report exists; skipped before that."""
    if not REPORT_MD.is_file():
        print("  (report not generated yet; its checks are skipped)")
        return

    text = REPORT_MD.read_text()

    for reference in set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)):
        path = (REPORT_MD.parent / reference).resolve()
        audit.check(path.is_file(), f"report references missing figure {reference}")

    match = re.search(
        r"<!-- analysis:start -->(.*?)<!-- analysis:end -->", text, re.S
    )
    if audit.check(match is not None,
                   "the analysis section is not delimited, so its length "
                   "cannot be checked against the brief's 400 to 600 words"):
        body = re.sub(r"[#*`_>|\[\]()-]", " ", match.group(1))
        words = len(body.split())
        low, high = WORD_RANGE
        audit.check(
            low <= words <= high,
            f"the analysis is {words} words; the brief asks for {low} to {high}",
        )
        print(f"  analysis length: {words} words")


def main() -> int:
    audit = Audit()
    results = load_runs()

    print("Auditing results/baseline.json and results/contrast.json")
    audit_noise(audit, results)
    audit_controlled_comparison(audit, results)
    audit_history(audit, results)
    audit_checkpoints(audit, results)
    audit_report(audit)

    return audit.report()


if __name__ == "__main__":
    raise SystemExit(main())
