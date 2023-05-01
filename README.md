# LowRank Adapters
This libary implements a series of [Low Rank Adapters](https://arxiv.org/abs/2106.09685) to apply to torch models in a simple manner without having to redefine your model. [Official Repository](https://github.com/microsoft/LoRA)

You can choose to only store the updated layers as shown below

```python
import timm 
import torch
from lora_adapters import LoraConv2d, apply_adapter, mark_only_lora_as_trainable, lora_state_dict
model = timm.create_model('resnet50', pretrained=True)
model = apply_adapter(model, LoraConv2d, rank=16)
model = mark_only_lora_as_trainable(model, bias='lora_only')
... Custom Training Loop ...
updates = lora_state_dict(model, bias='lora_only')
torch.save(updates, 'updates.ckpt')
```

```python
import timm 
import torch
from lora_adapters import LoraConv2d, apply_adapter, mark_lora_as_trainable, lora_state_dict
model = timm.create_model('resnet50', pretrained=True)
updates = torch.load('updates.ckpt')
model.load_state_dict(updates, strict=False)
```

Or you can reverse the lora adaptations and save the full state dictionary

```python
import timm 
import torch
from lora_adapters import LoraConv2d, apply_adapter, mark_only_lora_as_trainable, undo_lora
model = timm.create_model('resnet50', pretrained=True)
model = apply_adapter(model, LoraConv2d, rank=16)
model = mark_only_lora_as_trainable(model, bias='lora_only')
... Custom Training Loop ...
model = undo_lora(model)
torch.save(model.state_dict(), 'model.ckpt')
```



## Warnings
The functions that adapt the model change them in place, so the below code won't work as expected, `model` and `lora_model` will be the same

```python
import timm 
from lora_adapters import LoraConv2d, apply_adapter
model = timm.create_model('resnet50', pretrained=True)
lora_model = apply_adapter(model, LoraConv2d, rank=16)
```