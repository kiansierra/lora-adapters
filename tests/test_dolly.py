import pytest
import torch
from lora_adapters import (
    LoraEmbedding,
    LoraMergedLinear,
    apply_adapter,
    lora_state_dict,
    mark_only_lora_as_trainable,
    undo_lora,
)
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM


@pytest.mark.skip(reason="Dolly model is too large to download")
def test_dolly_initialization():
    input_tensor = torch.randint(0, 512, (1, 256)).to("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16
    )
    output = model(input_tensor).logits  # Output has to go before adapter because adapter changes model in place
    model = apply_adapter(model, LoraEmbedding, rank=16, regex_pattern=".*embed_in")
    model = apply_adapter(model, LoraMergedLinear, rank=16, regex_pattern=".*query_key_value")
    output_lora = model(input_tensor).logits
    assert torch.equal(output, output_lora), "Adapter returns different outputs than original model"


@pytest.mark.skip(reason="Dolly model is too large to download")
def test_dolly_conversion():
    model = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16
    )
    lora_model = apply_adapter(model, LoraEmbedding, rank=16)
    lora_model = apply_adapter(model, LoraMergedLinear, rank=16, enable_lora=[True])
    original_linear_modules = [module for module in lora_model.modules() if type(module) == nn.Linear]
    assert len(original_linear_modules) == 0, "Adapter Hasn't converted all Linear layers"
    original_emb_modules = [module for module in lora_model.modules() if type(module) == nn.Embedding]
    assert len(original_emb_modules) == 0, "Adapter Hasn't converted all Embedding layers"


@pytest.mark.skip(reason="Dolly model is too large to download")
@pytest.mark.parametrize("bias", ["all", "lora_only", "none"])
def test_dolly_training(bias):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", torch_dtype=torch.bfloat16).to(device)
    model = apply_adapter(model, LoraEmbedding, rank=16, regex_pattern=".*embed_in")
    model = apply_adapter(model, LoraMergedLinear, rank=16, regex_pattern=".*0.*query_key_value")
    model = mark_only_lora_as_trainable(model, bias=bias)
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=1e-3)
    inputs = torch.randint(0, 512, (1, 64)).to(device)
    for _ in range(2):
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = outputs.mean()
        loss.backward()
        optimizer.step()

    model.eval()
    output_lora = model(inputs).logits
    model = undo_lora(model)
    output = model(inputs).logits

    assert torch.equal(output, output_lora), "Adapter returns different outputs than original model after training"


@pytest.mark.skip(reason="Dolly model is too large to download")
@pytest.mark.parametrize("bias", ["all", "lora_only", "none"])
def test_dolly_updates(bias):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", torch_dtype=torch.bfloat16).to(device)
    model = apply_adapter(model, LoraEmbedding, rank=16, regex_pattern=".*embed_in")
    model = apply_adapter(model, LoraMergedLinear, rank=16, regex_pattern=".*0.*query_key_value")
    model = mark_only_lora_as_trainable(model, bias=bias)
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=1e-3)
    inputs = torch.randint(0, 512, (1, 64)).to(device)
    for _ in range(2):
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = outputs.mean()
        loss.backward()
        optimizer.step()

    model.eval()
    output_lora = model(inputs).logits
    updates = lora_state_dict(model, bias=bias)
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", torch_dtype=torch.bfloat16).to(device)
    loaded_keys = model.load_state_dict(updates, strict=False)
    assert len(loaded_keys.unexpected_keys) == 0, "Unexpected keys in state dict"
    output = model(inputs).logits

    assert torch.equal(
        output, output_lora
    ), "Adapter returns different outputs than original model after training and applying state dict updates"
