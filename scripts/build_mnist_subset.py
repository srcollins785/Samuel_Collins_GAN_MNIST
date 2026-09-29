"""Download MNIST and record the 10,000-image subset both runs train on.

The images are not committed - the brief forbids submitting dataset
downloads - so what goes into the repository is the manifest: the indices,
the seed that chose them, and a checksum over them. Running this on
another machine at the same seed reproduces the subset exactly, and
``validate_results.py`` checks both runs recorded the same checksum.

Usage
-----
    .venv/bin/python scripts/build_mnist_subset.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gan_mnist import RunConfig, data  # noqa: E402
from gan_mnist._config import MANIFEST_PATH  # noqa: E402


def main() -> int:
    config = RunConfig()
    print(f"Downloading MNIST into {data.DATA_DIR.relative_to(REPO_ROOT)}/ "
          f"(gitignored)")
    manifest = data.write_manifest(config)

    print(f"Population        {manifest['population']:,} training images")
    print(f"Subset            {manifest['subset_size']:,} at seed "
          f"{manifest['seed']}")
    print(f"Checksum          {manifest['checksum']}")
    print(f"Normalization     {manifest['normalization']}")
    print(f"Wrote             {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
