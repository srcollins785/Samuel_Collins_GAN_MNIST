"""Layer summaries: measured shapes, not declared ones."""

from gan_mnist import models, summary


def test_generator_summary_ends_at_the_image_size(config):
    both = summary.both(config)
    last = both["generator"]["layers"][-1]
    assert last["type"] == "Tanh"
    assert last["output_shape"] == (models.IMAGE_PIXELS,)


def test_discriminator_summary_ends_at_one_score(config):
    both = summary.both(config)
    last = both["discriminator"]["layers"][-1]
    assert last["type"] == "Linear"
    assert last["output_shape"] == (1,)


def test_summary_parameter_total_matches_the_model(config):
    generator, _ = models.build(config)
    both = summary.both(config)
    assert both["generator"]["parameters"]["total"] == \
        models.parameter_count(generator)["total"]


def test_every_layer_is_listed(config):
    both = summary.both(config)
    generator, discriminator = models.build(config)
    for key, module in (("generator", generator), ("discriminator", discriminator)):
        leaves = [n for n, m in module.named_modules() if n and not list(m.children())]
        assert len(both[key]["layers"]) == len(leaves)


def test_markdown_renders_a_table(config):
    text = summary.as_markdown(summary.both(config)["generator"])
    assert "| Layer | Type | Output shape | Parameters |" in text
    assert "Generator" in text
