import pytest
import timm
import torch
from torch import nn
from torch.optim import AdamW

from lora_adapters import LoraConv2d, apply_adapter, freeze_bn, lora_state_dict, mark_only_lora_as_trainable, undo_lora


def test_resnet50_initialization():
    input_tensor = torch.randn(1, 3, 224, 224)
    model = timm.create_model("resnet50", pretrained=False)
    output = model(input_tensor)  # Output has to go before adapter because adapter changes model in place
    lora_model = apply_adapter(model, LoraConv2d, rank=4)
    reversed_lora_model = undo_lora(lora_model)
    output = model(input_tensor)
    output_lora = reversed_lora_model(input_tensor)
    assert torch.equal(output, output_lora), "Adapter returns different outputs than original model"


def test_resnet50_conversion():
    model = timm.create_model("resnet50", pretrained=False)
    lora_model = apply_adapter(model, LoraConv2d, rank=4)
    original_conv_modules = [module for module in lora_model.modules() if type(module) == nn.Conv2d]
    assert len(original_conv_modules) == 0, "Adapter Hasn't converted all conv2d layers"


def test_resnet50_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("resnet50", pretrained=False).to(device)
    model = apply_adapter(model, LoraConv2d, rank=16)
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=1e-3)
    inputs = torch.randn(1, 3, 224, 224).to(device)
    targets = torch.randint(0, 1000, (1,)).to(device)
    for _ in range(4):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    output_lora = model(inputs)
    model = undo_lora(model)
    output = model(inputs)

    assert torch.equal(output, output_lora), "Adapter returns different outputs than original model after training"


@pytest.mark.parametrize("bias", ["all", "lora_only", "none"])
def test_resnet50_updates(bias):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("resnet50", pretrained=True).to(device)
    model = apply_adapter(model, LoraConv2d, rank=16)
    model = mark_only_lora_as_trainable(model, bias=bias)
    model = freeze_bn(model)
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=1e-3)
    inputs = torch.randn(1, 3, 224, 224).to(device)
    targets = torch.randint(0, 1000, (1,)).to(device)
    for _ in range(4):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    output_lora = model(inputs)
    updates = lora_state_dict(model, bias=bias)
    model = timm.create_model("resnet50", pretrained=True).to(device).eval()
    loaded_keys = model.load_state_dict(updates, strict=False)
    assert len(loaded_keys.unexpected_keys) == 0, "Unexpected keys in state dict"
    output = model(inputs)

    assert torch.equal(
        output, output_lora
    ), "Adapter returns different outputs than original model after training and applying state dict updates"
