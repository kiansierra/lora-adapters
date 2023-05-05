import pytest
import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModel, AutoModelForSequenceClassification

from lora_adapters import (
    LoraEmbedding,
    LoraLinear,
    apply_adapter,
    lora_state_dict,
    mark_only_lora_as_trainable,
    undo_lora,
)


def test_bert_initialization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randint(0, 512, (1, 256)).to(device)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5).to(device)
    output = model(input_tensor).logits  # Output has to go before adapter because adapter changes model in place
    model = apply_adapter(model, LoraEmbedding, rank=16, regex_pattern=".*word_embeddings")
    model = apply_adapter(model, LoraLinear, rank=16, regex_pattern=".*query")
    model = apply_adapter(model, LoraLinear, rank=16, regex_pattern=".*value")
    reversed_lora_model = undo_lora(model)
    output_lora = reversed_lora_model(input_tensor).logits
    assert torch.equal(output, output_lora), "Adapter returns different outputs than original model"


def test_bert_conversion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5).to(device)
    model = apply_adapter(model, LoraEmbedding, rank=16)
    model = apply_adapter(model, LoraLinear, rank=16)
    original_linear_modules = [module for module in model.modules() if type(module) == nn.Linear]
    assert len(original_linear_modules) == 0, "Adapter Hasn't converted all Linear layers"
    original_emb_modules = [module for module in model.modules() if type(module) == nn.Embedding]
    assert len(original_emb_modules) == 0, "Adapter Hasn't converted all Embedding layers"


def test_bert_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5).to(device)
    model = apply_adapter(model, LoraEmbedding, rank=16, regex_pattern=".*word_embeddings")
    model = apply_adapter(model, LoraLinear, rank=16, regex_pattern=".*query")
    model = apply_adapter(model, LoraLinear, rank=16, regex_pattern=".*value")
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=1e-3)
    inputs = torch.randint(0, 512, (1, 256)).to(device)
    targets = torch.randint(0, 5, (1,)).to(device)
    for _ in range(4):
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    output_lora = model(inputs).logits
    model = undo_lora(model)
    output = model(inputs).logits

    assert torch.equal(output, output_lora), "Adapter returns different outputs than original model after training"


@pytest.mark.parametrize("bias", ["all", "lora_only", "none"])
def test_bert_updates(bias):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("bert-base-uncased", num_labels=5).to(device)
    model = apply_adapter(model, LoraEmbedding, rank=16, regex_pattern=".*word_embeddings")
    model = apply_adapter(model, LoraLinear, rank=16, regex_pattern=".*query")
    model = apply_adapter(model, LoraLinear, rank=16, regex_pattern=".*value")
    model = mark_only_lora_as_trainable(model, bias=bias)
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=1e-3)
    inputs = torch.randint(0, 512, (1, 256)).to(device)
    for _ in range(4):
        optimizer.zero_grad()
        outputs = model(inputs).last_hidden_state
        loss = outputs.mean()
        loss.backward()
        optimizer.step()

    model.eval()
    output_lora = model(inputs).last_hidden_state
    updates = lora_state_dict(model, bias=bias)
    model = AutoModel.from_pretrained("bert-base-uncased", num_labels=5).to(device)
    loaded_keys = model.load_state_dict(updates, strict=False)
    assert len(loaded_keys.unexpected_keys) == 0, "Unexpected keys in state dict"
    output = model(inputs).last_hidden_state

    assert torch.equal(
        output, output_lora
    ), "Adapter returns different outputs than original model after training and applying state dict updates"
