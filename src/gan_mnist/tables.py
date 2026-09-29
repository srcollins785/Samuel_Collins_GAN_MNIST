"""Tables built from what the runs recorded.

Everything here takes saved run records and returns Markdown. Nothing
formats a number it was told; every value is read out of the result
dictionaries the training loop wrote, so a table cannot disagree with the
run it describes.
"""

NA = "N/A"


def fmt(number, places: int = 4, default: str = NA) -> str:
    if number is None:
        return default
    return f"{number:.{places}f}"


def fmt_seconds(seconds, default: str = NA) -> str:
    if seconds is None:
        return default
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)} m {remainder:.0f} s"


def settings_table(results: dict) -> str:
    """The compact settings-and-runtime table the brief asks for.

    Rows are the settings; columns are the runs. Laid out this way because
    the point of the table is that one row differs and the rest do not,
    which is visible across a row and invisible down a column.
    """
    names = list(results)
    rows = [
        ("Discriminator learning rate", lambda r: f"{r['config']['discriminator_lr']:.0e}"),
        ("Generator learning rate", lambda r: f"{r['config']['generator_lr']:.0e}"),
        ("Optimizer", lambda r: f"Adam, betas ({r['config']['beta1']}, {r['config']['beta2']})"),
        ("Random seed", lambda r: str(r["config"]["seed"])),
        ("Training images", lambda r: f"{r['config']['subset_size']:,}"),
        ("Batch size", lambda r: str(r["config"]["batch_size"])),
        ("Batches per epoch", lambda r: str(r["history"][0]["batches"]) if r["history"] else NA),
        ("Latent dimension", lambda r: str(r["config"]["latent_dim"])),
        ("Epochs", lambda r: str(r["config"]["epochs"])),
        ("Generator parameters", lambda r: f"{r['parameters']['generator']['total']:,}"),
        ("Discriminator parameters", lambda r: f"{r['parameters']['discriminator']['total']:,}"),
        ("Device", lambda r: r["device"]),
        ("Fixed noise checksum", lambda r: f"`{r['fixed_noise_checksum'][:12]}`"),
        ("Runtime", lambda r: fmt_seconds(r["total_seconds"])),
    ]

    header = "| Setting | " + " | ".join(n.capitalize() for n in names) + " |"
    divider = "|---" * (len(names) + 1) + "|"
    lines = [header, divider]
    for label, accessor in rows:
        cells = [accessor(results[n]) for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def history_table(history, caption_columns=None) -> str:
    """Per-epoch losses, accuracy and time for one run."""
    lines = [
        "| Epoch | Generator loss | Discriminator loss | "
        "D accuracy (real) | D accuracy (fake) | Time |",
        "|---|---|---|---|---|---|",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {fmt(row['generator_loss'])} | "
            f"{fmt(row['discriminator_loss'])} | "
            f"{fmt(row['discriminator_accuracy_real'], 3)} | "
            f"{fmt(row['discriminator_accuracy_fake'], 3)} | "
            f"{row['seconds']:.1f} s |"
        )
    lines.append("")
    return "\n".join(lines)


def checkpoint_table(results: dict) -> str:
    """Which grids exist, at which epochs, from which noise.

    The checksum column is the point: it is the same value in every row of
    both runs, which is what "the same sixteen vectors at every checkpoint"
    means operationally.
    """
    lines = [
        "| Run | Epoch | Label | Noise checksum | Output range |",
        "|---|---|---|---|---|",
    ]
    for name, result in results.items():
        for record in result["checkpoints"]:
            epoch = record["epoch"]
            label = "before training" if epoch == 0 else f"after epoch {epoch}"
            lines.append(
                f"| {name} | {epoch} | {label} | "
                f"`{record['noise_checksum'][:12]}` | "
                f"[{record['image_min']:.2f}, {record['image_max']:.2f}] |"
            )
    lines.append("")
    return "\n".join(lines)
