import pytest
import torch
from torch import nn

from loralib import LoraConv2d, LoraEmbedding, LoraLinear


@pytest.mark.parametrize("input_dim,bias", [[64, True], [128, False]])
def test_conv(input_dim, bias):

    layer = nn.Conv2d(input_dim, input_dim * 2, 3, padding=1, bias=bias)
    lora_layer = LoraConv2d(layer, rank=4)
    input_tensor = torch.randn(1, input_dim, 32, 32)
    lora_layer(input_tensor).sum().backward()
    assert lora_layer.original_weight.grad is None, "LoraConv2d has Grads in original weight"


@pytest.mark.parametrize("input_dim", [64, 128, 256])
def test_embeddings(input_dim):

    layer = nn.Embedding(100, input_dim, 3)
    lora_layer = LoraEmbedding(layer, rank=4).eval().train()
    input_tensor = torch.randint(0, 100, (1, 64))
    lora_layer(input_tensor).sum().backward()

    assert lora_layer.original_weight.grad is None, "LoraEmbedding has Grads in original weight"


@pytest.mark.parametrize("input_dim,bias", [[64, True], [128, False]])
def test_linear(input_dim, bias):

    layer = nn.Linear(input_dim, 3, bias=bias)
    lora_layer = LoraLinear(layer, rank=4)
    input_tensor = torch.randn(1, input_dim)
    lora_layer(input_tensor).sum().backward()

    assert lora_layer.original_weight.grad is None, "LoraLinear has Grads in original weight"
