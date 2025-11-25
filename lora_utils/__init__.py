"""
lora_utils - A utility library for LoRA (Low-Rank Adaptation) operations with PyTorch models.

This package provides tools for injecting, managing, and manipulating LoRA layers
in neural network models, particularly for fine-tuning large models efficiently.
"""

from .modules import LoRA
from .lora_utils import *

__version__ = "0.1.0"

__all__ = [
    "LoRA",
    "inject_init_lora_for_model",
    "inject_pretrained_lora_for_model",
    "marge_lora_and_weight",
    "separate_lora_from_model",
    "get_module_by_key",
    "inject_empty_lora_layer",
]
