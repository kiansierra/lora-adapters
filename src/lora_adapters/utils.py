import re
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from torch import nn
from typing_extensions import Literal

from .layers import LoraLayerType

__all__ = ["apply_adapter", "undo_lora", "mark_only_lora_as_trainable", "lora_state_dict", "freeze_bn"]

BiasTypes = Literal["none", "all", "lora_only"]


# pylint: disable=protected-access
def apply_adapter(
    model: nn.Module,
    adapter_class: LoraLayerType,
    rank: int,
    lora_alpha: float = 1.0,
    regex_pattern: str = ".*",
    name_list: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """Adapts the models layer to the adapter_class.
    Args:
        model: Model to be adapted.
        adapter_class: Adapter class to be applied to the models layers.
        lora_alpha: Scaling factor to apply to the Low Rank Adapter.
            Default: ``1.0``
        rank: Rank to adapt the layer.
            Default: ``None``
        regex_pattern: Regular expression to match the layers to be adapted.
        **kwargs: Additional arguments to be passed to the adapter class.
    Returns:
        Adapted Model
    """
    if len(model._modules) == 0:
        return model
    if name_list is None:
        name_list = []
    new_modules = OrderedDict()
    for name in model._modules:
        module = model._modules[name]
        module_name_list = name_list + [name]
        if isinstance(module, adapter_class.__bases__[0]) and re.match(regex_pattern, ".".join(module_name_list)):
            new_modules[name] = adapter_class(module, lora_alpha=lora_alpha, rank=rank, **kwargs)
        else:
            new_modules[name] = apply_adapter(
                module,
                adapter_class,
                lora_alpha=lora_alpha,
                rank=rank,
                regex_pattern=regex_pattern,
                name_list=module_name_list,
                **kwargs,
            )
        del module
    model._modules = new_modules
    return model


def undo_lora(model: nn.Module) -> nn.Module:
    """
    Args:
        model: Reveses LoRa adaptations and keeps new updated weights.
    Returns:
        Unadapted Model
    """
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


# pylint: enable=protected-access


def mark_only_lora_as_trainable(model: nn.Module, bias: BiasTypes = "none") -> nn.Module:
    """
    Args:
        model: Model to updated trainable layers.
        bias: Trainable Bias option to be applied to the model.
            Default: ``none``
    returns:
        Model with only the lora layers as trainable and the chosen biases.
    """
    if bias not in ["none", "all", "lora_only"]:
        raise ValueError(f"Unknown bias option {bias} chose one of none, all, lora_only")
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    if bias == "none":
        return model
    if bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
        return model
    if bias == "lora_only":
        for module in model.modules():
            if getattr(module, "is_lora", False) and hasattr(module, "bias") and module.bias is not None:
                module.bias.requires_grad = True
    return model


def lora_state_dict(model: nn.Module, bias: BiasTypes = "none") -> Dict[str, torch.Tensor]:
    """
    Args:
        model: Model to obtain the LoRa state dict.
        bias: Bias option to be saved from the model. Should be set to the same as in mark_only_lora_as_trainable.
            Default: ``none``
    returns:
        State Dict only containing LoRa layers weights and selected bias.
    """
    if bias not in ["none", "all", "lora_only"]:
        raise ValueError(f"Unknown bias option {bias} chose one of none, all, lora_only")
    lora_dict = {name: module for name, module in model.named_modules() if getattr(module, "is_lora", False)}
    lora_dict_weights = {f"{name}.weight": module.weight for name, module in lora_dict.items()}
    if bias == "none":
        return lora_dict_weights
    if bias == "all":
        bias_dict = {name: param for name, param in model.named_parameters() if name.split(".")[-1] == "bias"}
        return {**lora_dict_weights, **bias_dict}
    if bias == "lora_only":
        lora_dict_bias = {
            f"{name}.bias": module.bias
            for name, module in lora_dict.items()
            if hasattr(module, "bias") and module.bias is not None
        }
        return {**lora_dict_weights, **lora_dict_bias}


def freeze_bn(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Args:
        model: Model to obtain the LoRa state dict.
        bias: Bias option to be saved from the model. Should be set to the same as in mark_only_lora_as_trainable.
            Default: ``none``
    returns:
        State Dict only containing LoRa layers weights and selected bias.
    """
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, "weight"):
                module.weight.requires_grad_(False)
            if hasattr(module, "bias"):
                module.bias.requires_grad_(False)
            module.eval()
    return model
