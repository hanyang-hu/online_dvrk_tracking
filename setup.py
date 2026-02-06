"""Setup for Differentiable-Calibration package."""
import numpy as np
from setuptools import setup, find_packages


def setup_package() -> None:
    setup(
        packages=find_packages(exclude=["docs", "scripts"]),
        include_dirs=[np.get_include()],  # Include directory for numpy
    )


if __name__ == "__main__":
    setup_package()
