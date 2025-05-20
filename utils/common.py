import random, os, numpy as np, torch, yaml
from torchvision import transforms
from typing import Tuple

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

def load_yaml(path:str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_transforms(size:int)->Tuple:
    def _resize(img, mask=None):
        img = transforms.functional.resize(img, (size, size),
                                           interpolation=transforms.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = transforms.functional.resize(mask, (size, size),
                                                interpolation=transforms.InterpolationMode.NEAREST)
        return img, mask

    def train_t(img, mask):
        img, mask = _resize(img, mask)
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
        return img, mask

    def val_t(img, mask):
        img, mask = _resize(img, mask)
        return img, mask

    return train_t, val_t
