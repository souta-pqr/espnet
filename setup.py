#!/usr/bin/env python3

"""ESPnet setup script."""

import os

from setuptools import find_packages, setup

requirements = {
    "install": [
        "setuptools>=38.5.1,<74.0.0",
        "packaging",
        "configargparse>=1.2.1",
        "typeguard",
        "humanfriendly",
        "scipy>=1.4.1",
        "filelock",
        "librosa==0.9.2",
        "jamo==0.4.1",  # For kss
        "PyYAML>=5.1.2",
        "soundfile>=0.10.2",
        "h5py>=2.10.0",
        "kaldiio>=2.18.0",
        "torch>=1.11.0",
        "torch_complex",
        "nltk>=3.4.5",
        # fix CI error due to the use of deprecated aliases
        "numpy<1.24",
        # https://github.com/espnet/espnet/runs/6646737793?check_suite_focus=true#step:8:7651
        "protobuf",
        "hydra-core",
        "opt-einsum",
        "lightning",
        # ASR
        "sentencepiece==0.2.0",
        "ctc-segmentation>=1.6.6",
        # TTS
        "pyworld>=0.3.4",
        "pypinyin<=0.44.0",
        "espnet_tts_frontend",
        # ENH
        "ci_sdr",
        "fast-bss-eval==0.1.3",
        # SPK
        "asteroid_filterbanks==0.4.0",
        # UASR
        "editdistance",
        # fix CI error due to the use of deprecated functions
        # https://github.com/espnet/espnet/actions/runs/3174416926/jobs/5171182884#step:8:8419
        # https://importlib-metadata.readthedocs.io/en/latest/history.html#v5-0-0
        "importlib-metadata<5.0",
    ],
    # train: The modules invoked when training only.
    "train": [
        "matplotlib",
        "pillow==9.5.0",
        "wandb",
        "tensorboard>=1.14",
    ],
    # recipe: The modules actually are not invoked in the main module of espnet,
    #         but are invoked for the python scripts in each recipe
    "recipe": [
        "espnet_model_zoo",
        "gdown",
        "resampy",
        "pysptk>=0.2.1",
        "morfessor",  # for zeroth-korean
        "youtube_dl",  # for laborotv
        "nnmnkwii",
        "museval>=0.2.1",
        "pystoi>=0.2.2",
        "mir-eval>=0.6",
        "fastdtw",
        "nara_wpe>=0.0.5",
        "sacrebleu>=1.5.1",
        "praatio>=6,<7",  # for librispeech phoneme alignment
        "scikit-learn>=1.0.0",  # for HuBERT kmeans
    ],
    # all: The modules should be optionally installled due to some reason.
    #      Please consider moving them to "install" occasionally
    # NOTE(kamo): The modules in "train" and "recipe" are appended into "all"
    "all": [
        # NOTE(kamo): Append modules requiring specific pytorch version or torch>1.3.0
        "torchaudio",
        "torch_optimizer",
        "fairscale",
        "transformers",
        "evaluate",
    ],
    "setup": [
        "pytest-runner",
    ],
    "test": [
        "pytest>=7.0.0,<8.4.0",
        "pytest-timeouts>=1.2.1",
        "pytest-pythonpath>=0.7.3",
        "pytest-cov>=2.7.1",
        "hacking>=2.0.0",
        "mock>=2.0.0",
        "pycodestyle",
        "jsondiff>=2.0.0",
        "flake8>=3.7.8",
        "flake8-docstrings>=1.3.1",
        "black",
        "isort",
    ],
    "doc": [
        "Jinja2<3.1",
        "sphinx<9.0.0",
        "sphinx-rtd-theme>=0.2.4",
        "sphinx-argparse>=0.2.5",
        "commonmark==0.8.1",
        "myst-parser",
        "nbsphinx>=0.4.2",
        "sphinx-markdown-tables>=0.0.12",
        "jupyterlab<5",
        "sphinx-markdown-builder",
    ],
}
requirements["all"].extend(requirements["train"] + requirements["recipe"])
requirements["test"].extend(requirements["train"])

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "espnet", "version.txt")
with open(version_file, "r") as f:
    version = f.read().strip()
setup(
    name="espnet",
    version=version,
    url="http://github.com/espnet/espnet",
    author="Shinji Watanabe",
    author_email="shinjiw@ieee.org",
    description="ESPnet: end-to-end speech processing toolkit",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache Software License",
    packages=find_packages(include=["espnet*"]),
    package_data={"espnet": ["version.txt"]},
    # #448: "scripts" is inconvenient for developping because they are copied
    # scripts=get_all_scripts('espnet/bin'),
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    python_requires=">=3.7.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
