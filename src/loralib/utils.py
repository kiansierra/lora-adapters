import re
from collections import OrderedDict
from typing import Dict, List, Literal, Optional

import torch
from torch import nn

from .layers import LoraLayerType

__all__ = [
    "apply_adapter",
    "undo_lora",
    "mark_only_lora_as_trainable",
    "lora_state_dict",
]

BiasTypes = Literal["none", "all", "lora_only"]


def apply_adapter(
    model: nn.Module,
    adapter_class: LoraLayerType,
    rank: Optional[int] = None,
    fraccion: Optional[float] = None,
    regex_pattern: str = ".*",
    name_list: Optional[List[str]] = None,
) -> nn.Module:
    """
    model: nn.Module to apply the adapter
    adapter_class: LoraLayerType class to apply

    """
    if len(model._modules) == 0:
        return model
    if name_list is None:
        name_list = []
    new_modules = OrderedDict()
    for name in model._modules:
        module = model._modules[name]
        module_name_list = name_list + [name]
        if isinstance(module, adapter_class.__bases__[-1]) and re.match(regex_pattern, ".".join(module_name_list)):
            new_modules[name] = adapter_class(module, rank=rank, fraccion=fraccion)
        else:
            new_modules[name] = apply_adapter(
                module,
                adapter_class,
                rank=rank,
                fraccion=fraccion,
                regex_pattern=regex_pattern,
                name_list=module_name_list,
            )
        del module
    model._modules = new_modules
    return model


def undo_lora(model: nn.Module) -> nn.Module:
    if len(model._modules) == 0:
        return model
    new_modules = OrderedDict()
    for name in model._modules:
        module = model._modules[name]
        is_lora = getattr(module, "is_lora", None)
        if is_lora:
            new_modules[name] = module.to_regular()
            continue
        new_modules[name] = undo_lora(module)
    model._modules = new_modules
    return model


def mark_only_lora_as_trainable(model: nn.Module, bias: BiasTypes = "none") -> nn.Module:
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    if bias == "none":
        return model
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for module in model.modules():
            if getattr(module, "is_lora", False) and hasattr(module, "bias") and module.bias is not None:
                module.bias.requires_grad = True
    else:
        raise ValueError(f"Unknown bias option {bias} chose one of none, all, lora_only")

    return model


def lora_state_dict(model: nn.Module, bias: BiasTypes = "none") -> Dict[str, torch.Tensor]:
    lora_dict = {name: module for name, module in model.named_modules() if getattr(module, "is_lora", False)}
    lora_dict_weights = {f"{name}.weight": module.weight for name, module in lora_dict.items()}
    if bias == "none":
        return lora_dict_weights
    elif bias == "all":
        bias_dict = {name: param for name, param in model.named_parameters() if name.split(".")[-1] == "bias"}
        return {**lora_dict_weights, **bias_dict}
    elif bias == "lora_only":
        lora_dict_bias = {
            f"{name}.bias": module.bias
            for name, module in lora_dict.items()
            if hasattr(module, "bias") and module.bias is not None
        }
        return {**lora_dict_weights, **lora_dict_bias}
    else:
        raise ValueError(f"Unknown bias option {bias} chose one of none, all, lora_only")
