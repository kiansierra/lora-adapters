# -------------------------------------------------------------------------------------------
# Original Implementation https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# -------------------------------------------------------------------------------------------

import inspect
import math
from typing import List, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LoraEmbedding", "LoraLinear", "LoraMergedLinear", "LoraConv2d"]


class LoRALayer:
    is_lora = True

    def __init__(
        self,
        rank: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.rank = rank
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoraEmbedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        layer: nn.Embedding,
        rank: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
    ):
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Embedding).args if arg != "self"]
        self.input_kwargs = {k: v for k, v in layer.__dict__.items() if k in named_inputs}
        nn.Embedding.__init__(self, **self.input_kwargs)
        LoRALayer.__init__(self, rank=rank, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        self.weight.data = layer.weight.data.clone()
        # Actual trainable parameters
        if rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, self.input_kwargs["num_embeddings"])))
            self.lora_B = nn.Parameter(self.weight.new_zeros((self.input_kwargs["embedding_dim"], rank)))
            self.scaling = self.lora_alpha / self.rank
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    @property
    def weight_delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.rank > 0:
                    self.weight.data -= self.weight_delta
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.rank > 0:
                    self.weight.data += self.weight_delta
                self.merged = True
        return self

    def forward(self, input: torch.Tensor):
        if self.rank > 0 and not self.merged:
            result = nn.Embedding.forward(self, input)
            if self.rank > 0:
                after_A = F.embedding(
                    input,
                    self.lora_A.transpose(0, 1),
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, input)

    def to_regular(self) -> nn.Embedding:
        """
        Converts the LoraEmbedding layer to a regular Embedding layer
        """
        layer = nn.Embedding(**self.input_kwargs)
        layer.weight.data = self.weight.data
        if not self.merged:
            layer.weight.data += self.weight_delta
        return layer


class LoraLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        layer: nn.Linear,
        rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
    ):
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Linear).args if arg != "self"]
        self.input_kwargs = {k: v for k, v in layer.__dict__.items() if k in named_inputs}
        self.input_kwargs["bias"] = layer.bias is not None
        nn.Linear.__init__(self, **self.input_kwargs)
        LoRALayer.__init__(
            self, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.weight.data = layer.weight.data.clone()
        # Actual trainable parameters
        if rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, self.input_kwargs["in_features"])))
            self.lora_B = nn.Parameter(self.weight.new_zeros((self.input_kwargs["out_features"], rank)))
            self.scaling = self.lora_alpha / self.rank
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        if self.input_kwargs["bias"]:
            self.bias.data = layer.bias.data.clone()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    @property
    def weight_delta(self) -> torch.Tensor:
        return self.T(self.lora_B @ self.lora_A) * self.scaling

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.rank > 0:
                    self.weight.data -= self.weight_delta
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.rank > 0:
                    self.weight.data += self.weight_delta
                self.merged = True
        return self

    def forward(self, input: torch.Tensor):
        if self.rank > 0 and not self.merged:
            result = F.linear(input, self.T(self.weight), bias=self.bias)
            if self.rank > 0:
                result += (
                    self.lora_dropout(input) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
                ) * self.scaling
            return result
        else:
            return F.linear(input, self.T(self.weight), bias=self.bias)

    def to_regular(self) -> nn.Linear:
        """
        Converts the LoraLinear layer to a regular Linear layer
        """
        layer = nn.Linear(**self.input_kwargs)
        layer.weight.data = self.weight.data
        if not self.merged:
            layer.weight.data += self.weight_delta
        if self.input_kwargs["bias"]:
            layer.bias.data = self.bias.data
        return layer


class LoraMergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        layer: nn.Linear,
        rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [True, False, True],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
    ):
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Linear).args if arg != "self"]
        self.input_kwargs = {k: v for k, v in layer.__dict__.items() if k in named_inputs}
        self.input_kwargs["bias"] = layer.bias is not None
        nn.Linear.__init__(self, **self.input_kwargs)
        LoRALayer.__init__(
            self, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights
        )
        in_features = self.input_kwargs["in_features"]
        out_features = self.input_kwargs["out_features"]
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.weight.data = layer.weight.data.clone()
        # Actual trainable parameters
        if rank > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), rank))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.rank
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        if self.input_kwargs["bias"]:
            self.bias.data = layer.bias.data.clone()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, input):
        result = input.new_zeros((*input.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = input.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*input.shape[:-1], self.out_features))

    def T(self, w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    @property
    def weight_delta(self) -> torch.Tensor:
        delta_w = (
            F.conv1d(self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora))
            .squeeze(0)
            .transpose(0, 1)
        )
        return self.zero_pad(self.T(delta_w * self.scaling)).transpose(0, 1)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.rank > 0 and any(self.enable_lora):
                    self.weight.data -= self.weight_delta
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.rank > 0 and any(self.enable_lora):
                    self.weight.data += self.weight_delta
                self.merged = True

    def forward(self, input: torch.Tensor):
        if self.merged:
            return F.linear(input, self.T(self.weight), bias=self.bias)
        else:
            result = F.linear(input, self.T(self.weight), bias=self.bias)
            if self.rank > 0:
                after_A = F.linear(self.lora_dropout(input), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result

    def to_regular(self) -> nn.Linear:
        """
        Converts the LoraMergedLinear layer to a regular Linear layer
        """
        layer = nn.Linear(**self.input_kwargs)
        layer.weight.data = self.weight.data
        if not self.merged:
            layer.weight.data += self.weight_delta
        if self.input_kwargs["bias"]:
            layer.bias.data = self.bias.data
        return layer


class LoraConv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        layer: nn.Conv2d,
        rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ):
        named_inputs = [arg for arg in inspect.getfullargspec(nn.Conv2d).args if arg != "self"]
        self.input_kwargs = {k: v for k, v in layer.__dict__.items() if k in named_inputs}
        self.input_kwargs["bias"] = layer.bias is not None
        nn.Conv2d.__init__(self, **self.input_kwargs)
        LoRALayer.__init__(
            self, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights
        )
        # Actual trainable parameters
        self.weight.data = layer.weight.data.clone()
        assert (
            self.input_kwargs["kernel_size"][0] == self.input_kwargs["kernel_size"][1]
        ), f"Only square kernels {self.input_kwargs['kernel_size']}"
        if rank > 0:
            kernel_size = self.input_kwargs["kernel_size"][0]
            in_channels = self.input_kwargs["in_channels"]
            out_channels = self.input_kwargs["out_channels"]
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, rank * kernel_size)))
            self.scaling = self.lora_alpha / self.rank
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if self.input_kwargs["bias"]:
            self.bias.data = layer.bias.data.clone()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @property
    def weight_delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= self.weight_delta
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.weight.data += self.weight_delta
                self.merged = True
        return self

    def forward(self, input: torch.Tensor):
        if self.rank > 0 and not self.merged:
            return F.conv2d(
                input,
                self.weight + self.weight_delta,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return nn.Conv2d.forward(self, input)

    def to_regular(self) -> nn.Conv2d:
        """
        Converts the LoraConv2d layer to a regular Conv2d layer
        """
        layer = nn.Conv2d(**self.input_kwargs)
        layer.weight.data = self.weight.data
        if not self.merged:
            layer.weight.data += self.weight_delta
        if self.input_kwargs["bias"]:
            layer.bias.data = self.bias.data
        return layer


LoraEmbeddingType = Type[LoraEmbedding]
LoraLinearType = Type[LoraLinear]
LoraConv2dType = Type[LoraConv2d]
LoraMergedLinearType = Type[LoraMergedLinear]
LoraLayerType = Union[LoraEmbeddingType, LoraLinearType, LoraConv2dType, LoraMergedLinearType]
