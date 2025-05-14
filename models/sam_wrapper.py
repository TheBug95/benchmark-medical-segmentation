"""
Lightweight wrapper for Segment Anything family.
Requires: pip install "segment-anything" and downloading model checkpoints.
"""

import torch, warnings
from pathlib import Path

def build_sam_model(name:str, num_classes:int):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        warnings.warn("segment-anything is not installed; returning dummy network")
        import torch.nn as nn
        return nn.Identity(), 1024

    ckpt_map = {
        "sam1":"sam_vit_b_01ec64.pth",
        "sam2":"sam_vit_l_0b3195.pth",
        "mobilesam":"mobile_sam.pt",
        "medsam":"medsam_vit_b.pth"
    }
    if name not in ckpt_map:
        raise ValueError(f"Unknown SAM variant {name}")
    ckpt_path = Path('weights')/ckpt_map[name]
    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt_path))
    # freeze encoder, keep mask decoder head trainable
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    # make last conv output num_classes masks
    sam.mask_decoder.output_hypernetworks_mlps = torch.nn.ModuleList(
        [torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, num_classes)
        ) for _ in range(4)]
    )
    return sam, 1024
