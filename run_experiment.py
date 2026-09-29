"""Train the GAN. One entry point for both experiments.

    .venv/bin/python run_experiment.py --run both

The baseline and the contrast run differ in the discriminator learning
rate and nothing else: the contrast configuration is derived from the
baseline by ``RunConfig.contrast()``, which changes that one field, so a
second difference cannot be introduced by editing the wrong line.

Results are written as files and nothing downstream imports this module.
The report reads ``results/``; the notebook calls the same package
functions this does. Both can therefore be rebuilt without retraining.
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from gan_mnist import (  # noqa: E402
    RunConfig, data, evaluate, models, plots, resolve_device, summary, train,
)
from gan_mnist._config import PLOTS_DIR, RESULTS_DIR  # noqa: E402


def relative(path: Path) -> str:
    """Repo-relative when the path is inside it, absolute otherwise.

    Scouting runs write outside the repository, and the report references
    figures by their repo-relative path. Failing on a scouting path would
    make the develop-time runs harder to do than the real ones.
    """
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def hardware() -> dict:
    """What the runtime figures in the report are runtimes on."""
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        chip = platform.processor() or "unknown"
    return {
        "chip": chip,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }


def run_one(config: RunConfig, dataset, device, results_dir: Path,
            plots_dir: Path) -> dict:
    """Train one configuration and write everything it produced."""
    loader = data.make_loader(dataset, config)
    fixed_noise = evaluate.make_fixed_noise(config)

    def on_checkpoint(epoch, images):
        path = plots_dir / f"{config.name}_grid_epoch{epoch}.png"
        plots.save_grid(images, path, plots.checkpoint_title(config.name, epoch))
        return {"grid": relative(path)}

    print(f"\n=== {config.name}: discriminator lr {config.discriminator_lr:.0e}, "
          f"generator lr {config.generator_lr:.0e}, device {device}")
    result = train.train(config, loader, device, fixed_noise,
                         on_checkpoint=on_checkpoint)

    result["hardware"] = hardware()
    result["started_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["subset_checksum"] = data.read_manifest()["checksum"]

    plots.save_loss_curves(
        result["history"], plots_dir / f"{config.name}_losses.png",
        f"Average loss per epoch - {config.name} run "
        f"(D lr {config.discriminator_lr:.0e})",
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{config.name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"  wrote {relative(path)}  "
          f"({result['total_seconds']:.1f}s total)")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="both",
                        choices=["baseline", "contrast", "both"])
    parser.add_argument("--contrast-lr", type=float, default=2e-5,
                        help="discriminator learning rate for the contrast run")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--generator-lr", type=float, default=None)
    parser.add_argument("--baseline-lr", type=float, default=None,
                        help="discriminator learning rate for the baseline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name-suffix", default="",
                        help="appended to run names, for scouting runs")
    parser.add_argument("--results-dir", default=None,
                        help="defaults to results/; point elsewhere for scouting")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--figures-only", action="store_true",
                        help="rebuild the comparison figures from saved results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    plots_dir = Path(args.plots_dir) if args.plots_dir else PLOTS_DIR

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
        overrides["checkpoints"] = tuple(
            c for c in RunConfig().checkpoints if c <= args.epochs
        )
    if args.subset_size is not None:
        overrides["subset_size"] = args.subset_size
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.generator_lr is not None:
        overrides["generator_lr"] = args.generator_lr
    if args.baseline_lr is not None:
        overrides["discriminator_lr"] = args.baseline_lr

    baseline = RunConfig(name=f"baseline{args.name_suffix}", **overrides)
    contrast = baseline.contrast(args.contrast_lr,
                                 name=f"contrast{args.name_suffix}")

    if args.figures_only:
        return rebuild_figures(results_dir, plots_dir)

    device = resolve_device(args.device)
    dataset, manifest = data.load_subset(baseline)
    print(f"Subset {manifest['subset_size']:,} images, "
          f"checksum {manifest['checksum'][:12]}")

    # The sixteen real examples the brief asks to be displayed.
    plots.save_real_grid(dataset, plots_dir / "real_samples.png",
                         seed=baseline.seed)

    # Layer summaries, written once: they do not depend on the run.
    (results_dir).mkdir(parents=True, exist_ok=True)
    (results_dir / "architecture.json").write_text(
        json.dumps(summary.both(baseline), indent=2, default=str) + "\n"
    )

    chosen = {"baseline": [baseline], "contrast": [contrast],
              "both": [baseline, contrast]}[args.run]
    for config in chosen:
        run_one(config, dataset, device, results_dir, plots_dir)

    if args.run == "both":
        rebuild_figures(results_dir, plots_dir, suffix=args.name_suffix)
    return 0


def rebuild_figures(results_dir: Path, plots_dir: Path, suffix: str = "") -> int:
    """The side-by-side figures, from saved results only."""
    histories = {}
    for name in ("baseline", "contrast"):
        path = results_dir / f"{name}{suffix}.json"
        if path.is_file():
            histories[name] = json.loads(path.read_text())["history"]

    if len(histories) < 2:
        print("Both runs are needed for the comparison figures; skipping.")
        return 0

    plots.save_comparison_curves(
        histories, plots_dir / "comparison_losses.png",
        "Discriminator learning-rate comparison",
    )
    plots.save_accuracy_curves(
        histories, plots_dir / "comparison_accuracy.png",
        "Discriminator accuracy per epoch",
    )
    print(f"  wrote comparison figures to {plots_dir.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
