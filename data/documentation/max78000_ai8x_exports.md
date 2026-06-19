# MAX78000 PyTorch Model Interface

Submit a PyTorch model definition file as `model_path`. The harness will import the
file, instantiate the model, train/evaluate it, and create the temporary ai8x
checkpoint for `ai8xize.py`.

The file must expose one no-argument factory or class:

- `build_model()`
- `create_model()`
- `get_model()`
- `Model`
- `Net`

Use ordinary `torch` and `torch.nn` modules. Do not import the MAX78000 toolchain or
call `ai8xize.py` from the sandbox.

The synthesis check maps the model `state_dict()` into an ai8x checkpoint. Name
convolution modules so their state dict keys line up with the YAML layer order. A
simple supported pattern is:

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 16)
        self.conv2 = ConvBlock(16, 16)
        self.conv3 = nn.Module()
        self.conv3.conv2d = nn.Conv2d(16, 8, 1)
```

This yields keys such as `conv1.conv2d.weight`, `conv2.conv2d.weight`, and
`conv3.conv2d.weight`. The YAML should contain matching layers in the same order.

For the COCO segmentation task, return a tensor shaped like `B x C x H x W`. The
train check treats the first output channel as a binary foreground mask and reports
`mAP50-95` from mask IoU thresholds. Use at least 8 output channels for MAX78000
segmentation-style outputs because the generated unload path is more reliable than
single-channel output.

Supported layer types for simple submissions:

- `nn.Conv2d`
- `nn.ConvTranspose2d`
- `nn.MaxPool2d`
- `torch.relu` / `torch.nn.functional.relu`
- `torch.cat` for skip connections, when reflected by the YAML `in_sequences`

Keep models small: the MAX78000 CNN accelerator has 64 processors, 512 KiB data
memory, 432 KiB kernel/weight memory, and 2 KiB bias memory.
