"""Packaging metadata for the ``gac`` reference implementation."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gac",
    version="0.1.0",
    description=(
        "Guided Adaptive Controller: noise-aware adaptive mixing for hybrid "
        "SFT-RL post-training of large language models (EMNLP 2026)."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yuelin Hu, Wei Liu, Zhenbo Yu, Zhengxue Cheng, Li Song",
    url="https://github.com/deepnovacore/GAC",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(exclude=("tests", "tests.*", "configs")),
    install_requires=[
        "torch>=2.1",
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
