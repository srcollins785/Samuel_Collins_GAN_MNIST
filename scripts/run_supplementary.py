"""Evidence beyond the one comparison the brief requires.

Two things the required comparison cannot do on its own, both cheap enough
here that leaving them undone would be a choice rather than a constraint:

**The sweep.** Baseline against contrast is two points. Two points cannot
distinguish "lowering the discriminator rate hurt" from "the relationship
is monotone in this direction", so the same experiment is run at four
rates and the trend is reported.

**The replication.** One seed cannot separate the effect of the learning
rate from the noise of initialization and batch order. Each arm is
repeated at three seeds, and the spread between seeds is reported beside
the difference between arms, so the reader can see which is larger.

Neither is the graded comparison, and the report labels both supplementary.
They exist so the analysis can say what it does and does not know, with a
number attached.

Usage
-----
    .venv/bin/python scripts/run_supplementary.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gan_mnist import (  # noqa: E402
    RunConfig, data, evaluate, plots, resolve_device, train,
)
from gan_mnist._config import PLOTS_DIR, RESULTS_DIR  # noqa: E402

SWEEP_RATES = (2e-5, 2e-4, 1e-3, 4e-3)
SEEDS = (42, 43, 44)
ARMS = {"baseline": 2e-4, "contrast": 2e-5}

SUPPLEMENTARY = RESULTS_DIR / "supplementary"
SWEEP_PLOTS = PLOTS_DIR / "supplementary"


def final(history: list) -> dict:
    last = history[-1]
    return {
        "generator_loss": last["generator_loss"],
        "discriminator_loss": last["discriminator_loss"],
        "discriminator_accuracy": last["discriminator_accuracy"],
    }


def run(config, dataset, device, plots_dir=None, grid_name=None) -> dict:
    loader = data.make_loader(dataset, config)
    noise = evaluate.make_fixed_noise(config)

    def on_checkpoint(epoch, images):
        if plots_dir is None or grid_name is None or epoch != config.epochs:
            return None
        path = plots_dir / f"{grid_name}.png"
        plots.save_grid(images, path, plots.checkpoint_title(grid_name, epoch))
        return {"grid": str(path.relative_to(REPO_ROOT))}

    return train.train(config, loader, device, noise,
                       on_checkpoint=on_checkpoint, log=lambda *a: None)


def sweep(dataset, device) -> list:
    """The same experiment at four discriminator rates, one seed."""
    rows = []
    print("Sweep - discriminator learning rate, seed 42")
    for rate in SWEEP_RATES:
        config = RunConfig(name=f"sweep_{rate:.0e}").contrast(
            rate, name=f"sweep_{rate:.0e}"
        )
        result = run(config, dataset, device, SWEEP_PLOTS, f"sweep_{rate:.0e}")
        row = {"discriminator_lr": rate, "seconds": result["total_seconds"],
               **final(result["history"])}
        rows.append(row)
        print(f"  {rate:.0e}   G {row['generator_loss']:.4f}   "
              f"D {row['discriminator_loss']:.4f}   "
              f"D-acc {row['discriminator_accuracy']:.3f}")
    return rows


def replication(dataset, device) -> list:
    """Both arms at three seeds each."""
    rows = []
    print("\nReplication - each arm at three seeds")
    for arm, rate in ARMS.items():
        for seed in SEEDS:
            config = RunConfig(name=f"{arm}_seed{seed}", seed=seed,
                               discriminator_lr=rate)
            result = run(config, dataset, device)
            row = {"arm": arm, "discriminator_lr": rate, "seed": seed,
                   **final(result["history"])}
            rows.append(row)
            print(f"  {arm:9s} seed {seed}   "
                  f"G {row['generator_loss']:.4f}   "
                  f"D-acc {row['discriminator_accuracy']:.3f}")
    return rows


def spread(rows: list) -> dict:
    """Between-seed spread against between-arm difference.

    The comparison is only informative if the gap between the arms is
    larger than the scatter within them. This computes both so the report
    does not have to assert which won.
    """
    summary = {}
    for arm in ARMS:
        values = [r["generator_loss"] for r in rows if r["arm"] == arm]
        summary[arm] = {
            "generator_loss_mean": sum(values) / len(values),
            "generator_loss_min": min(values),
            "generator_loss_max": max(values),
            "generator_loss_range": max(values) - min(values),
            "seeds": len(values),
        }
    summary["between_arm_difference"] = abs(
        summary["baseline"]["generator_loss_mean"]
        - summary["contrast"]["generator_loss_mean"]
    )
    summary["largest_within_arm_range"] = max(
        summary[arm]["generator_loss_range"] for arm in ARMS
    )
    summary["separated"] = (
        summary["between_arm_difference"] > summary["largest_within_arm_range"]
    )
    return summary


def main() -> int:
    SUPPLEMENTARY.mkdir(parents=True, exist_ok=True)
    SWEEP_PLOTS.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    dataset, manifest = data.load_subset(RunConfig())
    print(f"Subset {manifest['subset_size']:,} images, "
          f"checksum {manifest['checksum'][:12]}, device {device}\n")

    sweep_rows = sweep(dataset, device)
    replication_rows = replication(dataset, device)
    summary = spread(replication_rows)

    payload = {
        "note": "Supplementary. Not the comparison the brief requires; see "
                "results/baseline.json and results/contrast.json for that.",
        "subset_checksum": manifest["checksum"],
        "device": str(device),
        "sweep": sweep_rows,
        "replication": replication_rows,
        "replication_summary": summary,
    }
    path = SUPPLEMENTARY / "supplementary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nBetween-arm difference in final generator loss: "
          f"{summary['between_arm_difference']:.4f}")
    print(f"Largest within-arm range across seeds:          "
          f"{summary['largest_within_arm_range']:.4f}")
    print(f"Arms separated beyond seed noise:               "
          f"{summary['separated']}")
    print(f"\nWrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
