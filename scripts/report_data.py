"""Load what the runs wrote, and answer questions about it.

The report is not allowed to state a number it did not read from disk, and
it is not allowed to import the package. This module is the boundary: it
reads ``results/*.json`` and ``data/subset_manifest.json`` and exposes
small query functions - which run ended with the lower generator loss, by
how much, whether the arms separate beyond seed noise - so the prose can
ask a question and get the measured answer rather than a remembered one.

That indirection is the point. Every superlative in the generated document
comes from a function here, so if the runs are repeated and the numbers
move, the sentences move with them, and a claim cannot outlive the result
that justified it.

``gan_mnist.tables`` formats the same kinds of table for the notebook,
which legitimately imports the package. The small overlap is deliberate:
the report keeps its own path to the numbers so that it stays rebuildable
from files alone.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
REPORT_DIR = REPO_ROOT / "report"

NA = "N/A"
RUNS = ("baseline", "contrast")


def load() -> dict:
    """Everything the report is built from. Missing pieces are None."""
    data = {"runs": {}}
    for name in RUNS:
        path = RESULTS_DIR / f"{name}.json"
        if not path.is_file():
            raise SystemExit(
                f"Missing {path.relative_to(REPO_ROOT)}. Run the experiments "
                f"first:\n  .venv/bin/python run_experiment.py --run both"
            )
        data["runs"][name] = json.loads(path.read_text())

    architecture = RESULTS_DIR / "architecture.json"
    data["architecture"] = (
        json.loads(architecture.read_text()) if architecture.is_file() else None
    )

    supplementary = RESULTS_DIR / "supplementary" / "supplementary.json"
    data["supplementary"] = (
        json.loads(supplementary.read_text()) if supplementary.is_file() else None
    )

    manifest = REPO_ROOT / "data" / "subset_manifest.json"
    data["manifest"] = json.loads(manifest.read_text()) if manifest.is_file() else None
    return data


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(number, places: int = 4, default: str = NA) -> str:
    if number is None:
        return default
    return f"{number:.{places}f}"


def rate(value) -> str:
    """Learning rates as the report states them, not as floats print."""
    return f"{value:.0e}".replace("e-0", "e-")


def seconds(value, default: str = NA) -> str:
    if value is None:
        return default
    if value < 60:
        return f"{value:.1f} s"
    minutes, remainder = divmod(value, 60)
    return f"{int(minutes)} m {remainder:.0f} s"


def figure(path: str, caption: str) -> str:
    """A figure reference relative to report/, where the Markdown lives."""
    return f"![{caption}](../{path})\n\n*{caption}*\n"


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def final(data: dict, run: str, field: str):
    history = data["runs"][run]["history"]
    return history[-1][field] if history else None


def at_epoch(data: dict, run: str, epoch: int, field: str):
    for row in data["runs"][run]["history"]:
        if row["epoch"] == epoch:
            return row[field]
    return None


def config(data: dict, run: str, field: str):
    return data["runs"][run]["config"][field]


def lower_final_generator_loss(data: dict) -> str:
    """Which run ended with the lower generator loss. Measured, not assumed."""
    return min(RUNS, key=lambda r: final(data, r, "generator_loss"))


def generator_loss_gap(data: dict) -> float:
    losses = [final(data, r, "generator_loss") for r in RUNS]
    return abs(losses[0] - losses[1])


def accuracy_gap(data: dict) -> float:
    values = [final(data, r, "discriminator_accuracy") for r in RUNS]
    return abs(values[0] - values[1])


def total_runtime(data: dict) -> float:
    return sum(data["runs"][r]["total_seconds"] for r in RUNS)


def generator_updates(data: dict, run: str = "baseline") -> int:
    result = data["runs"][run]
    return result["history"][0]["batches"] * result["config"]["epochs"]


def checkpoint_grid(data: dict, run: str, epoch: int):
    for record in data["runs"][run]["checkpoints"]:
        if record["epoch"] == epoch:
            return record.get("grid")
    return None


def sweep_rows(data: dict) -> list:
    return (data["supplementary"] or {}).get("sweep", [])


def sweep_best_quality_claim(data: dict) -> dict:
    """The sweep's extremes, by loss. Quality is judged visually, not here."""
    rows = sweep_rows(data)
    if not rows:
        return {}
    return {
        "lowest_generator_loss": min(rows, key=lambda r: r["generator_loss"]),
        "highest_generator_loss": max(rows, key=lambda r: r["generator_loss"]),
        "rates": [r["discriminator_lr"] for r in rows],
    }


def replication(data: dict) -> dict:
    return (data["supplementary"] or {}).get("replication_summary", {})


def arms_separated(data: dict) -> bool:
    return bool(replication(data).get("separated"))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def settings_table(data: dict) -> str:
    """Settings down the rows, runs across the columns.

    Laid out this way because the point is that one row differs and the
    rest do not, and that is visible along a row.
    """
    rows = [
        ("Discriminator learning rate",
         lambda r: f"**{rate(r['config']['discriminator_lr'])}**"),
        ("Generator learning rate", lambda r: rate(r["config"]["generator_lr"])),
        ("Optimizer",
         lambda r: f"Adam ({r['config']['beta1']}, {r['config']['beta2']})"),
        ("Random seed", lambda r: str(r["config"]["seed"])),
        ("Training images", lambda r: f"{r['config']['subset_size']:,}"),
        ("Subset checksum", lambda r: f"`{r['subset_checksum'][:12]}`"),
        ("Batch size", lambda r: str(r["config"]["batch_size"])),
        ("Batches per epoch", lambda r: str(r["history"][0]["batches"])),
        ("Epochs", lambda r: str(r["config"]["epochs"])),
        ("Generator updates",
         lambda r: f"{r['history'][0]['batches'] * r['config']['epochs']:,}"),
        ("Latent dimension", lambda r: str(r["config"]["latent_dim"])),
        ("Evaluation noise checksum",
         lambda r: f"`{r['fixed_noise_checksum'][:12]}`"),
        ("Device", lambda r: r["device"]),
        ("Runtime", lambda r: seconds(r["total_seconds"])),
        ("Final generator loss",
         lambda r: fmt(r["history"][-1]["generator_loss"])),
        ("Final discriminator loss",
         lambda r: fmt(r["history"][-1]["discriminator_loss"])),
        ("Final discriminator accuracy",
         lambda r: fmt(r["history"][-1]["discriminator_accuracy"], 3)),
    ]
    lines = ["| Setting | Baseline | Contrast |", "|---|---|---|"]
    for label, accessor in rows:
        cells = [accessor(data["runs"][name]) for name in RUNS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def history_table(data: dict, run: str) -> str:
    lines = [
        "| Epoch | Generator loss | Discriminator loss | D accuracy (real) | "
        "D accuracy (fake) | D accuracy | Time |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in data["runs"][run]["history"]:
        lines.append(
            f"| {row['epoch']} | {fmt(row['generator_loss'])} | "
            f"{fmt(row['discriminator_loss'])} | "
            f"{fmt(row['discriminator_accuracy_real'], 3)} | "
            f"{fmt(row['discriminator_accuracy_fake'], 3)} | "
            f"{fmt(row['discriminator_accuracy'], 3)} | "
            f"{row['seconds']:.1f} s |"
        )
    lines.append("")
    return "\n".join(lines)


def architecture_table(data: dict, which: str) -> str:
    summary = (data["architecture"] or {}).get(which)
    if not summary:
        return "_Architecture summary not generated._\n"

    counts = summary["parameters"]
    lines = [
        f"**{summary['name']}** - input `{tuple(summary['input_shape'])}`, "
        f"{counts['total']:,} parameters, all trainable",
        "",
        "| Layer | Type | Output shape | Parameters |",
        "|---|---|---|---|",
    ]
    for row in summary["layers"]:
        shape = tuple(row["output_shape"])
        lines.append(
            f"| `{row['layer']}` | {row['type']} | `{shape}` | "
            f"{row['parameters']:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def sweep_table(data: dict) -> str:
    rows = sweep_rows(data)
    if not rows:
        return "_Supplementary sweep not run._\n"
    lines = [
        "| Discriminator lr | Generator loss | Discriminator loss | "
        "D accuracy |",
        "|---|---|---|---|",
    ]
    for row in rows:
        marker = ""
        for name in RUNS:
            if abs(row["discriminator_lr"]
                   - config(data, name, "discriminator_lr")) < 1e-12:
                marker = f" ({name})"
        lines.append(
            f"| {rate(row['discriminator_lr'])}{marker} | "
            f"{fmt(row['generator_loss'])} | "
            f"{fmt(row['discriminator_loss'])} | "
            f"{fmt(row['discriminator_accuracy'], 3)} |"
        )
    lines.append("")
    return "\n".join(lines)


def replication_table(data: dict) -> str:
    rows = (data["supplementary"] or {}).get("replication", [])
    if not rows:
        return "_Supplementary replication not run._\n"
    lines = [
        "| Arm | Discriminator lr | Seed | Generator loss | D accuracy |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {rate(row['discriminator_lr'])} | {row['seed']} | "
            f"{fmt(row['generator_loss'])} | "
            f"{fmt(row['discriminator_accuracy'], 3)} |"
        )
    lines.append("")
    return "\n".join(lines)
