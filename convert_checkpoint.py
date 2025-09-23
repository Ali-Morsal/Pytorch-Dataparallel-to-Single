import torch
import os


def remove_dataparallel_prefix(checkpoint_path, output_path=None):
    """
    Converts a checkpoint trained with nn.DataParallel to a single GPU/CPU compatible checkpoint.

    Args:
        checkpoint_path (str): Path to the original .pth or .pt checkpoint
        output_path (str, optional): Path to save the new checkpoint.
                                     If None, will append '_unparalleled' to the original name.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Determine if it's a full checkpoint dict or just state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict_key = 'state_dict'
    elif isinstance(checkpoint, dict) and all(k.startswith('module.') for k in checkpoint.keys()):
        state_dict_key = None  # checkpoint itself is state_dict
    else:
        state_dict_key = None  # checkpoint itself is state_dict

    if state_dict_key:
        state_dict = checkpoint[state_dict_key]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    # Save new checkpoint
    if state_dict_key:
        checkpoint[state_dict_key] = new_state_dict
        new_checkpoint = checkpoint
    else:
        new_checkpoint = new_state_dict

    if output_path is None:
        base, ext = os.path.splitext(checkpoint_path)
        output_path = f"{base}_unparalleled{ext}"

    torch.save(new_checkpoint, output_path)
    print(f"Converted checkpoint saved to: {output_path}")


# ====== Write Checkpoint Path here ======
checkpoint_path = "./best_dense_unet.pth"
remove_dataparallel_prefix(checkpoint_path)
