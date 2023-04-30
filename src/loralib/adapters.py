import re
from collections import OrderedDict
from typing import List, Optional

from torch import nn

from .layers import LoraLayerType

__all__ = ['apply_adapter', 'undo_lora']

def apply_adapter(model:nn.Module,
                  adapter_class: LoraLayerType, 
                  rank: Optional[int] = None,
                  fraccion: Optional[float] = None,
                  regex_pattern: str = ".*",
                  name_list : Optional[List[str]] = None):
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
            new_modules[name] = apply_adapter(module, adapter_class, rank=rank, fraccion=fraccion,regex_pattern=regex_pattern, name_list=module_name_list)
        del module
    model._modules = new_modules
    return model

def undo_lora(model):
    if len(model._modules) == 0:
        return model
    new_modules = OrderedDict()
    for name in model._modules:
        module = model._modules[name]
        is_lora = getattr(module, 'is_lora',  None)
        if is_lora:
            new_modules[name] = module.to_regular()
            continue        
        new_modules[name] = undo_lora(module)
    model._modules = new_modules
    return model
        