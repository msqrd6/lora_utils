from setuptools import setup, find_packages

setup(
    name="lora-utils",
    version="0.1.3",
    author="msqrd6",
    description="A utility library for LoRA (Low-Rank Adaptation) operations with PyTorch models",
    url="https://github.com/msqrd6/lora_utils",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
    ],
    license="MIT",
)
