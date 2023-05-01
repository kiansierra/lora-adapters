import inspect
from typing import Optional, Type, Union

import numpy as np
import torch
from torch import Tensor, nn

__all__ = ["LoraLayerType", "LoraConv2d", "LoraLinear", "LoraEmbedding"]


class LoraLinear(nn.Linear):
    """Implementation of Low Rank Adapters (https://arxiv.org/abs/2106.09685) on a Linear Layer.
    Args:
        layer: Original layer to be adapted.
        lora_alpha: Scaling factor to apply to the Low Rank Adapter.
            Default: ``1``
        rank: Rank to adapt the layer.
            Default: ``None``
        frac: Fraccion of the orignal layer dimension to establish the rank of the Adapter.
            Default: ``None``
    Returns:
        Adapted layer
    """

    is_lora = True

    def __init__(
        self, layer: nn.Linear, lora_alpha: float = 1, rank: Optional[int] = None, fraccion: Optional[float] = None
    ):
        if rank is None and fraccion is None:
            raise ValueError("Rank and fraccion can't be None at the same time")
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Linear).args if arg != "self"]
        self.input_kwargs = {k: v for k, v in layer.__dict__.items() if k in named_inputs}
        self.input_kwargs["bias"] = layer.bias is not None
        super().__init__(**self.input_kwargs)
        del self.weight
        if fraccion is not None:
            rank = int(fraccion * layer.weight.size(1))
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        device = layer.weight.device
        self.original_weight = nn.Parameter(layer.weight.data.clone(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty((layer.weight.size(0), rank), device=device), requires_grad=True)
        self.lora_B = nn.Parameter(torch.empty((rank, layer.weight.size(1)), device=device), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if self.input_kwargs["bias"]:
            self.bias = nn.Parameter(layer.bias.data.clone(), requires_grad=False)

    @property
    def weight_delta(self) -> Tensor:
        return torch.mm(self.lora_A, self.lora_B) * self.scaling

    @property
    def weight(self) -> Tensor:
        return self.original_weight + self.weight_delta

    def to_regular(self) -> nn.Linear:
        """
        Converts the LoraLinear layer to a regular Linear layer
        """
        layer = nn.Linear(**self.input_kwargs)
        layer.weight.data = self.weight
        layer.bias.data = self.bias.data
        return layer


class LoraEmbedding(nn.Embedding):
    """Implementation of Low Rank Adapters (https://arxiv.org/abs/2106.09685) on a Embedding Layer.
    Args:
        layer: Original layer to be adapted.
        lora_alpha: Scaling factor to apply to the Low Rank Adapter.
            Default: ``1``
        rank: Rank to adapt the layer.
            Default: ``None``
        frac: Fraccion of the orignal layer dimension to establish the rank of the Adapter.
            Default: ``None``
    Returns:
        Adapted layer
    """

    is_lora = True

    def __init__(
        self, layer: nn.Embedding, lora_alpha: float = 1, rank: Optional[int] = None, fraccion: Optional[float] = None
    ):
        if rank is None and fraccion is None:
            raise ValueError("Rank and fraccion can't be None at the same time")
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Embedding).args if arg != "self"]
        self.input_kwargs = {k: v for k, v in layer.__dict__.items() if k in named_inputs}
        super().__init__(**self.input_kwargs)
        del self.weight
        if fraccion is not None:
            rank = int(fraccion * layer.weight.size(1))
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        device = layer.weight.device
        self.original_weight = nn.Parameter(layer.weight.data.clone(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty((layer.weight.size(0), rank), device=device), requires_grad=True)
        self.lora_B = nn.Parameter(torch.empty((rank, *layer.weight.shape[1:]), device=device), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight_delta(self) -> Tensor:
        return self.lora_A @ self.lora_B * self.scaling

    @property
    def weight(self) -> Tensor:
        return self.original_weight + self.weight_delta

    def to_regular(self) -> nn.Embedding:
        """
        Converts the LoraEmbedding layer to a regular Embedding layer
        """
        layer = nn.Embedding(**self.input_kwargs)
        layer.weight.data = self.weight
        return layer


class LoraConv2d(nn.Conv2d):
    """Implementation of Low Rank Adapters (https://arxiv.org/abs/2106.09685) on a Conv2d Layer.
    Args:
        layer: Original layer to be adapted.
        lora_alpha: Scaling factor to apply to the Low Rank Adapter.
            Default: ``1``
        rank: Rank to adapt the layer.
            Default: ``None``
        frac: Fraccion of the orignal layer dimension to establish the rank of the Adapter.
            Default: ``None``
    Returns:
        Adapted layer
    """

    is_lora = True

    def __init__(
        self, layer: nn.Conv2d, lora_alpha: float = 1, rank: Optional[int] = None, fraccion: Optional[float] = None
    ):
        if rank is None and fraccion is None:
            raise ValueError("Rank and fraccion can't be None at the same time")
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Conv2d).args if arg != "self"]
        self.input_kwargs = {k: v for k, v in layer.__dict__.items() if k in named_inputs}
        self.input_kwargs["bias"] = layer.bias is not None
        super().__init__(**self.input_kwargs)
        del self.weight
        if fraccion is not None:
            rank = int(fraccion * layer.weight.size(1))
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        device = layer.weight.device
        self.original_weight = nn.Parameter(layer.weight.data.clone(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty((layer.weight.size(0), rank), device=device), requires_grad=True)
        self.lora_B = nn.Parameter(torch.empty((rank, *layer.weight.shape[1:]), device=device), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if self.input_kwargs["bias"]:
            self.bias = nn.Parameter(layer.bias.data.clone(), requires_grad=False)

    @property
    def weight_delta(self) -> Tensor:
        """
        i -> in_features
        r -> rank
        o -> out_features
        k -> kernel_size_1
        j -> kernel_size_2
        """
        return torch.einsum("ir, rokj -> iokj", self.lora_A, self.lora_B) * self.scaling

    @property
    def weight(self) -> Tensor:
        return self.original_weight + self.weight_delta

    def to_regular(self) -> nn.Conv2d:
        """
        Converts the LoraConv2d layer to a regular Conv2d layer
        """
        layer = nn.Conv2d(**self.input_kwargs)
        layer.weight.data = self.weight
        if self.input_kwargs["bias"]:
            layer.bias.data = self.bias.data
        return layer


LoraEmbeddingType = Type[LoraEmbedding]
LoraLinearType = Type[LoraLinear]
LoraConv2dType = Type[LoraConv2d]
LoraLayerType = Union[LoraEmbeddingType, LoraLinearType, LoraConv2dType]
