import numpy as np, torch
from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class CocoSegmentationDataset(Dataset):
    def __init__(self, images_dir, ann_file, transform=None, return_masks=True):
        self.images_dir = Path(images_dir)
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.return_masks = return_masks

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.images_dir / info['file_name']).convert('RGB')
        if self.return_masks:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)
            mask = np.zeros((info['height'], info['width']), dtype=np.uint8)
            for a in anns:
                mask |= self.coco.annToMask(a)
            mask = Image.fromarray(mask)
        if self.transform:
            if self.return_masks:
                img, mask = self.transform(img, mask)
            else:
                img = self.transform(img, None)[0]
        if self.return_masks:
            return TF.to_tensor(img), torch.as_tensor(np.array(mask), dtype=torch.long)
        return TF.to_tensor(img), img_id
