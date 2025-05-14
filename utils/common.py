import random, os, numpy as np, torch, yaml
from torchvision import transforms
from typing import Tuple

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True

def load_yaml(path:str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_transforms(size:int)->Tuple:
    """Return (train_tf, val_tf) that act jointly on image & mask"""
    def _joint_resize(img, mask=None):
        img = transforms.functional.resize(img, (size, size),
                                           interpolation=transforms.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = transforms.functional.resize(mask, (size, size),
                                                interpolation=transforms.InterpolationMode.NEAREST)
        return img, mask

    def train_transform(img, mask):
        img, mask = _joint_resize(img, mask)
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
        return img, mask

    def val_transform(img, mask):
        img, mask = _joint_resize(img, mask)
        return img, mask

    return train_transform, val_transform
