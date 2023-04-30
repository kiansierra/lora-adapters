import pytest
import torch
from loralib import LoraConv2d, LoraEmbedding, LoraLinear
from torch import nn


@pytest.mark.parametrize("input_dim,bias", [[64 , True], [128, False]])
def test_conv(input_dim, bias):
    
    layer = nn.Conv2d(input_dim, input_dim*2, 3, padding=1, bias=bias)
    lora_layer = LoraConv2d(layer, rank=4)
    input_tensor = torch.randn(1, input_dim, 32, 32)
    assert torch.equal(layer(input_tensor), lora_layer(input_tensor)), "LoraConv2d initialization is returning different outputs"
    
    
@pytest.mark.parametrize("input_dim", [64,128,256])
def test_embeddings(input_dim):
    
    layer = nn.Embedding(100, input_dim, 3)
    lora_layer = LoraEmbedding(layer, rank=4)
    input_tensor = torch.randint(0, 100, (1, 64))
    assert torch.equal(layer(input_tensor), lora_layer(input_tensor)), "LoraEmbedding initialization is returning different outputs"
    
    
@pytest.mark.parametrize("input_dim,bias", [[64 , True], [128, False]])
def test_linear(input_dim, bias):
    
    layer = nn.Linear(input_dim, 3, bias=bias)
    lora_layer = LoraLinear(layer, rank=4)
    input_tensor = torch.randn(1, input_dim)
    assert torch.equal(layer(input_tensor), lora_layer(input_tensor)), "LoraLinear initialization is returning different outputs"
    
    
    

    

    

