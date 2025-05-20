import torch, warnings
from pathlib import Path

def build_sam_model(name:str, num_classes:int):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        warnings.warn("segment-anything not installed; returning Identity")
        return torch.nn.Identity(), 1024
    ckpt = {
        "sam1":"sam_vit_b_01ec64.pth",
        "sam2":"sam_vit_l_0b3195.pth",
        "mobilesam":"mobile_sam.pt",
        "medsam":"medsam_vit_b.pth"
    }[name]
    sam = sam_model_registry["vit_b"](checkpoint=str(Path('weights')/ckpt))
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    return sam, 1024
