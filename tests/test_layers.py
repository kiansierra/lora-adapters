import pytest
import torch
from torch import nn

from lora_adapters import LoraConv2d, LoraEmbedding, LoraLinear, LoraMergedLinear


@pytest.mark.parametrize("input_dim,bias", [[64, True], [128, False]])
def test_conv(input_dim, bias):
    input_tensor = torch.randn(1, input_dim, 32, 32)
    layer = nn.Conv2d(input_dim, input_dim * 2, 3, padding=1, bias=bias)
    output = layer(input_tensor)  # Output has to go before adapter because adapter changes model in place
    lora_layer = LoraConv2d(layer, rank=4)
    assert torch.equal(output, lora_layer(input_tensor)), "LoraConv2d initialization is returning different outputs"


@pytest.mark.parametrize("input_dim", [64, 128, 256])
def test_embeddings(input_dim):
    input_tensor = torch.randint(0, 100, (1, 64))
    layer = nn.Embedding(100, input_dim, 3)
    output = layer(input_tensor)  # Output has to go before adapter because adapter changes model in place
    lora_layer = LoraEmbedding(layer, rank=4)
    assert torch.equal(output, lora_layer(input_tensor)), "LoraEmbedding initialization is returning different outputs"


@pytest.mark.parametrize("input_dim,bias", [[64, True], [128, False]])
def test_linear(input_dim, bias):
    input_tensor = torch.randn(1, input_dim)
    layer = nn.Linear(input_dim, 3, bias=bias)
    output = layer(input_tensor)  # Output has to go before adapter because adapter changes model in place
    lora_layer = LoraLinear(layer, rank=4)
    assert torch.equal(output, lora_layer(input_tensor)), "LoraLinear initialization is returning different outputs"


@pytest.mark.parametrize("input_dim,bias", [[64, True], [128, False]])
def test_merged_linear(input_dim, bias):
    input_tensor = torch.randn(1, input_dim)
    layer = nn.Linear(input_dim, 3 * 32, bias=bias)
    output = layer(input_tensor)  # Output has to go before adapter because adapter changes model in place
    lora_layer = LoraMergedLinear(layer, rank=4, enable_lora=[True, False, True])
    assert torch.equal(
        output, lora_layer(input_tensor)
    ), "LoraMergedLinear initialization is returning different outputs"
