"""Table formatting, and the one column the comparison depends on."""

from gan_mnist import tables


def _result(name, d_lr):
    return {
        "config": {
            "name": name, "seed": 42, "subset_size": 10000, "batch_size": 128,
            "latent_dim": 100, "epochs": 5, "generator_lr": 2e-4,
            "discriminator_lr": d_lr, "beta1": 0.5, "beta2": 0.999,
        },
        "device": "mps",
        "fixed_noise_checksum": "abcdef0123456789" * 4,
        "history": [{
            "epoch": 1, "generator_loss": 1.2345, "discriminator_loss": 0.9876,
            "discriminator_accuracy_real": 0.81, "discriminator_accuracy_fake": 0.77,
            "discriminator_accuracy": 0.79, "batches": 78, "seconds": 12.3,
        }],
        "checkpoints": [
            {"epoch": 0, "noise_checksum": "abcdef0123456789" * 4,
             "image_min": -0.9, "image_max": 0.9},
        ],
        "total_seconds": 61.5,
        "parameters": {
            "generator": {"total": 1489424, "trainable": 1489424, "frozen": 0},
            "discriminator": {"total": 533505, "trainable": 533505, "frozen": 0},
        },
    }


def test_settings_table_shows_both_learning_rates():
    table = tables.settings_table({
        "baseline": _result("baseline", 2e-4),
        "contrast": _result("contrast", 1e-3),
    })
    assert "Discriminator learning rate" in table
    assert "2e-04" in table
    assert "1e-03" in table


def test_settings_table_has_a_column_per_run():
    table = tables.settings_table({
        "baseline": _result("baseline", 2e-4),
        "contrast": _result("contrast", 1e-3),
    })
    header = table.splitlines()[0]
    assert header.count("|") == 4


def test_history_table_has_a_row_per_epoch():
    history = _result("baseline", 2e-4)["history"]
    table = tables.history_table(history)
    body = [line for line in table.splitlines() if line.startswith("| 1 ")]
    assert len(body) == 1


def test_checkpoint_table_reports_the_shared_checksum():
    table = tables.checkpoint_table({"baseline": _result("baseline", 2e-4)})
    assert "abcdef012345" in table
    assert "before training" in table


def test_seconds_format_switches_to_minutes():
    assert tables.fmt_seconds(45.2) == "45.2 s"
    assert tables.fmt_seconds(125.0) == "2 m 5 s"


def test_missing_values_render_as_na():
    assert tables.fmt(None) == "N/A"
    assert tables.fmt_seconds(None) == "N/A"
