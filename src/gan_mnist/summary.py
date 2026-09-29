"""Layer summaries and tensor shapes.

The brief asks for both to be shown. torch has no ``model.summary()``, so
this walks the module and records the shape each layer actually produces
by running one batch through with forward hooks - the shapes are measured,
not derived from the constructor arguments, so a layer that silently
reshapes cannot be misreported.
"""

import torch
from torch import nn

from . import models


def summarize(module: nn.Module, input_shape: tuple, batch_size: int = 2) -> dict:
    """Run one batch through and record what each leaf layer emitted.

    ``batch_size`` is 2 rather than 1 because BatchNorm1d raises on a
    single-sample batch in training mode. Nothing here depends on the
    batch size; it is dropped from the reported shapes.
    """
    rows = []
    handles = []

    def hook(name, layer):
        def record(_module, _inputs, output):
            rows.append({
                "layer": name,
                "type": type(_module).__name__,
                "output_shape": tuple(output.shape[1:]),
                "parameters": sum(p.numel() for p in _module.parameters()),
            })
        return record

    for name, layer in module.named_modules():
        if name and not list(layer.children()):
            handles.append(layer.register_forward_hook(hook(name, layer)))

    was_training = module.training
    module.eval()
    # BatchNorm in eval mode uses running statistics, so a 2-sample batch
    # is safe and the recorded shapes are the ones inference produces.
    with torch.no_grad():
        module(torch.zeros(batch_size, *input_shape))
    module.train(was_training)

    for handle in handles:
        handle.remove()

    counts = models.parameter_count(module)
    return {
        "name": type(module).__name__,
        "input_shape": input_shape,
        "layers": rows,
        "parameters": counts,
    }


def as_markdown(summary: dict) -> str:
    """The summary as a Markdown table, for the report and the notebook."""
    lines = [
        f"**{summary['name']}** - input {tuple(summary['input_shape'])}, "
        f"{summary['parameters']['total']:,} parameters "
        f"({summary['parameters']['trainable']:,} trainable)",
        "",
        "| Layer | Type | Output shape | Parameters |",
        "|---|---|---|---|",
    ]
    for row in summary["layers"]:
        lines.append(
            f"| `{row['layer']}` | {row['type']} | "
            f"{tuple(row['output_shape'])} | {row['parameters']:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def both(config) -> dict:
    """Summaries of the generator and discriminator as they are built."""
    generator, discriminator = models.build(config)
    return {
        "generator": summarize(generator, (config.latent_dim,)),
        "discriminator": summarize(discriminator, models.IMAGE_SHAPE),
    }
