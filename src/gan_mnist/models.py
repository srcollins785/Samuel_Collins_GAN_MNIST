"""The two networks, and why they are shaped the way they are.

Both are small fully connected stacks, which the brief permits. The
generator widens from the latent vector to the image; the discriminator
narrows from the image to a single score. Neither takes a digit label:
this is an unconditional GAN, and the label is not an input anywhere in
this repository.

The discriminator returns a **logit**, not a probability. There is no
final sigmoid, and the loss is ``BCEWithLogitsLoss``, which folds the
sigmoid into the loss and computes it in log-space. The brief asks that
the binary cross-entropy formulation be consistent with whether the
discriminator returns logits or probabilities; pairing a raw logit with
``BCEWithLogitsLoss`` is that consistency, and it avoids the saturation
that a separate sigmoid followed by a log would produce once the
discriminator becomes confident.
"""

import torch
from torch import nn

IMAGE_SHAPE = (1, 28, 28)
IMAGE_PIXELS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]

# The slope the DCGAN paper uses, and the value carried over from the
# TensorFlow tutorial this work adapts.
LEAK = 0.2


class Generator(nn.Module):
    """Noise in, image out.

    Input is a latent vector of independent standard normals - it carries
    no information about digits, and everything the samples eventually
    show is learned in these weights. Output is a 1x28x28 image in
    [-1, 1], matching the normalization the real images were given.

    BatchNorm on the two hidden layers after the first: it keeps the
    activations from collapsing early in training, which is when this
    generator is most fragile. The output layer has no normalization,
    because tanh already bounds it.
    """

    def __init__(self, latent_dim: int = 100, image_shape=IMAGE_SHAPE):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        pixels = image_shape[0] * image_shape[1] * image_shape[2]

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(LEAK, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(LEAK, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(LEAK, inplace=True),

            nn.Linear(1024, pixels),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.net(z)
        return flat.view(z.size(0), *self.image_shape)


class Discriminator(nn.Module):
    """Image in, one unbounded score out.

    The target is a real/generated decision, not a digit class. Dropout
    on both hidden layers: without it this discriminator memorizes a
    10,000-image subset quickly, wins outright, and stops producing a
    useful gradient for the generator.
    """

    def __init__(self, image_shape=IMAGE_SHAPE, dropout: float = 0.3):
        super().__init__()
        self.image_shape = image_shape
        pixels = image_shape[0] * image_shape[1] * image_shape[2]

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(pixels, 512),
            nn.LeakyReLU(LEAK, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LeakyReLU(LEAK, inplace=True),
            nn.Dropout(dropout),

            # No sigmoid. See the module docstring.
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build(config) -> tuple:
    """Both networks, on the resolved device, seeded so runs are comparable.

    Seeding here rather than at import time means the contrast run starts
    from the same initial weights as the baseline, which the brief
    requires: "initial random seed" is one of the things held fixed.
    """
    torch.manual_seed(config.seed)
    generator = Generator(latent_dim=config.latent_dim)
    discriminator = Discriminator()
    return generator, discriminator


def parameter_count(module: nn.Module) -> dict:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
