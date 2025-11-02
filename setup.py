from pathlib import Path

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).resolve().parent
README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()

setup(
    name="bayesian-kan-experiments",
    version="0.1.0",
    description="Kolmogorov–Arnold Networks with Bayesian inference backends.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="",
    packages=find_packages(exclude=("examples", "tests")),
    python_requires=">=3.9",
    install_requires=REQUIREMENTS,
    include_package_data=True,
)
