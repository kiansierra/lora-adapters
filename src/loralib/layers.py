
import inspect
from typing import Optional, Type, Union

import numpy as np
import torch
from torch import nn

__all__ = ['LoraLayerType', 'LoraConv2d', 'LoraLinear', 'LoraEmbedding']

class LoraLinear(nn.Linear):
    
    is_lora = True
    
    def __init__(self,  layer: nn.Conv2d, rank: Optional[int] = None, fraccion: Optional[float] = None):
        if rank is None and fraccion is None:
            raise ValueError("Rank and fraccion can't be None at the same time")
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Linear).args if arg != 'self']
        self.input_kwargs = {k:v for k,v in layer.__dict__.items() if k in named_inputs}
        self.input_kwargs['bias'] = layer.bias is not None
        super().__init__(**self.input_kwargs)
        del self.weight
        if fraccion is not None:
            rank = int(fraccion * layer.weight.size(1))
        self.original_weight = nn.Parameter(layer.weight.data.clone(), requires_grad=False)
        self.lora_a = nn.Parameter(torch.empty(layer.weight.size(0), rank), requires_grad=True)
        self.lora_b = nn.Parameter(torch.empty(rank, layer.weight.size(1)), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b)
        if self.input_kwargs['bias']:
            self.bias = nn.Parameter(layer.bias.data.clone(), requires_grad=False)
    
    @property
    def weight(self):
        return self.original_weight + torch.mm(self.lora_a, self.lora_b)
    
    def to_regular(self) -> nn.Linear: 
        layer = nn.Linear(**self.input_kwargs)
        layer.weight.data = self.weight.data
        layer.bias.data = self.bias.data
        return layer

class LoraEmbedding(nn.Embedding):
    """
    Creates a LoraEmbedding layer
    """
    
    is_lora = True
    
    def __init__(self, layer: nn.Embedding, rank: Optional[int] = None, fraccion: Optional[float] = None):
        if rank is None and fraccion is None:
            raise ValueError("Rank and fraccion can't be None at the same time")
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Embedding).args if arg != 'self']
        self.input_kwargs = {k:v for k,v in layer.__dict__.items() if k in named_inputs}
        super().__init__(**self.input_kwargs)
        del self.weight
        if fraccion is not None:
            rank = int(fraccion * layer.weight.size(1))
        self.original_weight = nn.Parameter(layer.weight.data.clone(), requires_grad=False)
        self.lora_a = nn.Parameter(torch.empty(layer.weight.size(0), rank), requires_grad=True)
        self.lora_b = nn.Parameter(torch.empty((rank, *layer.weight.shape[1:])), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b)
    
    @property
    def weight(self):
        """
        i -> in_features
        r -> rank
        o -> out_features
        k -> kernel_size_1
        j -> kernel_size_2
        """
        return self.original_weight +  self.lora_A @ self.lora_B
    
    
    def to_regular(self) -> nn.Embedding: 
        """
        Converts the LoraEmbedding layer to a regular Embedding layer
        """
        layer = nn.Embedding(**self.input_kwargs)
        layer.weight.data = self.weight.data
        return layer 
    
class LoraConv2d(nn.Conv2d):
    
    is_lora = True
    
    def __init__(self, layer: nn.Conv2d, rank: Optional[int] = None, fraccion: Optional[float] = None):
        if rank is None and fraccion is None:
            raise ValueError("Rank and fraccion can't be None at the same time")
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Conv2d).args if arg != 'self']
        self.input_kwargs = {k:v for k,v in layer.__dict__.items() if k in named_inputs}
        self.input_kwargs['bias'] = layer.bias is not None
        super().__init__(**self.input_kwargs)
        del self.weight
        if fraccion is not None:
            rank = int(fraccion * layer.weight.size(1))
        self.original_weight = nn.Parameter(layer.weight.data.clone(), requires_grad=False)
        self.lora_a = nn.Parameter(torch.empty(layer.weight.size(0), rank), requires_grad=True)
        self.lora_b = nn.Parameter(torch.empty((rank, *layer.weight.shape[1:])), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b)
        if self.input_kwargs['bias']:
            self.bias = nn.Parameter(layer.bias.data.clone(), requires_grad=False)
    
    @property
    def weight(self):
        """
        i -> in_features
        r -> rank
        o -> out_features
        k -> kernel_size_1
        j -> kernel_size_2
        """
        return self.original_weight +  torch.einsum('ir, rokj -> iokj', self.lora_a, self.lora_b)
    
    
    def to_regular(self) -> nn.Conv2d: 
        layer = nn.Conv2d(**self.input_kwargs)
        layer.weight.data = self.weight.data
        if self.input_kwargs['bias']:
            layer.bias.data = self.bias.data
        return layer
    
class LoraEmbedding(nn.Embedding):
    
    is_lora = True
    
    def __init__(self, layer: nn.Embedding, rank: Optional[int] = None, fraccion: Optional[float] = None):
        if rank is None and fraccion is None:
            raise ValueError("Rank and fraccion can't be None at the same time")
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Embedding).args if arg != 'self']
        self.input_kwargs = {k:v for k,v in layer.__dict__.items() if k in named_inputs}
        super().__init__(**self.input_kwargs)
        del self.weight
        if fraccion is not None:
            rank = int(fraccion * layer.weight.size(1))
        self.original_weight = nn.Parameter(layer.weight.data.clone(), requires_grad=False)
        self.lora_a = nn.Parameter(torch.empty(layer.weight.size(0), rank), requires_grad=True)
        self.lora_b = nn.Parameter(torch.empty((rank, *layer.weight.shape[1:])), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b)
    
    @property
    def weight(self):
        """
        i -> in_features
        r -> rank
        o -> out_features
        k -> kernel_size_1
        j -> kernel_size_2
        """
        return self.original_weight +  self.lora_a @ self.lora_b
    
    
    def to_regular(self) -> nn.Embedding: 
        layer = nn.Embedding(**self.input_kwargs)
        layer.weight.data = self.weight.data
        return layer
    
    
LoraEmbeddingType = Type[LoraEmbedding]
LoraLinearType = Type[LoraLinear]
LoraConv2dType = Type[LoraConv2d]

LoraLayerType = Union[LoraEmbeddingType, LoraLinearType, LoraConv2dType]
