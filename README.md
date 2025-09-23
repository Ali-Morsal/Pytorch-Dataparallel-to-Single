# Convert DataParallel Checkpoints to Single GPU

This small Python script converts PyTorch checkpoints trained with `nn.DataParallel` (multi-GPU) 
to be compatible with a single GPU or CPU setup.

Many PyTorch users face issues when loading multi-GPU checkpoints on a single device, 
because the state_dict keys are prefixed with `module.`. This script removes that prefix 
and saves a new checkpoint that works on one GPU.

## Features

- Supports `.pth` and `.pt` files
- Automatically detects if checkpoint contains `state_dict` or full model
- Saves a new checkpoint with `_unparalleled` appended to the original filename
- Easy to reuse on any checkpoint

## Usage

1. Clone or download this script.
2. Edit the `checkpoint_path` variable inside the script to point to your checkpoint file.
3. Run the script:

```bash
python convert_checkpoint.py
