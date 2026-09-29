"""Write configuration.yaml from what the runs actually recorded.

The brief asks that the seed, library versions, hardware, subset size,
batch size, latent dimension and optimizer settings be recorded. Writing
that file by hand would make it a description of what was intended; this
generates it from results/*.json, so it is a description of what ran.

Usage
-----
    .venv/bin/python scripts/write_configuration.py
"""

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from gan_mnist._config import RESULTS_DIR  # noqa: E402

OUTPUT = REPO_ROOT / "configuration.yaml"


def main() -> int:
    runs = {}
    for name in ("baseline", "contrast"):
        path = RESULTS_DIR / f"{name}.json"
        if not path.is_file():
            raise SystemExit(f"Missing {path.relative_to(REPO_ROOT)}")
        runs[name] = json.loads(path.read_text())

    manifest = json.loads((REPO_ROOT / "data" / "subset_manifest.json").read_text())
    baseline = runs["baseline"]
    hardware = baseline["hardware"]

    document = {
        "generated": date.today().isoformat(),
        "generated_by": "scripts/write_configuration.py, from results/*.json",
        "assignment": "Lab Assignment 5: Generative Adversarial Networks",
        "data": {
            "dataset": manifest["dataset"],
            "population": manifest["population"],
            "subset_size": manifest["subset_size"],
            "selection": manifest["selection"],
            "selection_seed": manifest["seed"],
            "subset_checksum": manifest["checksum"],
            "image_shape": manifest["image_shape"],
            "normalization": manifest["normalization"],
            "labels_used": False,
        },
        "model": {
            "type": "fully connected, unconditional",
            "latent_dim": baseline["config"]["latent_dim"],
            "generator_parameters": baseline["parameters"]["generator"]["total"],
            "discriminator_parameters": baseline["parameters"]["discriminator"]["total"],
            "discriminator_output": "logit (no sigmoid)",
            "loss": "BCEWithLogitsLoss",
        },
        "training": {
            "seed": baseline["config"]["seed"],
            "epochs": baseline["config"]["epochs"],
            "batch_size": baseline["config"]["batch_size"],
            "batches_per_epoch": baseline["history"][0]["batches"],
            "generator_updates": (baseline["history"][0]["batches"]
                                  * baseline["config"]["epochs"]),
            "optimizer": "Adam",
            "betas": [baseline["config"]["beta1"], baseline["config"]["beta2"]],
            "generator_lr": baseline["config"]["generator_lr"],
            "checkpoints": baseline["config"]["checkpoints"],
            "evaluation_noise_vectors": baseline["config"]["n_eval_samples"],
            "evaluation_noise_checksum": baseline["fixed_noise_checksum"],
        },
        "experiments": {
            name: {
                "discriminator_lr": result["config"]["discriminator_lr"],
                "runtime_seconds": round(result["total_seconds"], 2),
                "device": result["device"],
                "final_generator_loss": round(
                    result["history"][-1]["generator_loss"], 4),
                "final_discriminator_loss": round(
                    result["history"][-1]["discriminator_loss"], 4),
                "final_discriminator_accuracy": round(
                    result["history"][-1]["discriminator_accuracy"], 4),
            }
            for name, result in runs.items()
        },
        "hardware": {
            "chip": hardware["chip"],
            "platform": hardware["platform"],
            "device_used": baseline["device"],
            "mps_available": hardware["mps_available"],
            "cuda_available": hardware["cuda_available"],
        },
        "software": {
            "python": hardware["python"],
            "torch": hardware["torch"],
        },
    }

    OUTPUT.write_text(yaml.safe_dump(document, sort_keys=False, width=88))
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
