import torch, numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datasets.coco_segmentation import CocoSegmentationDataset
from models.model_zoo import get_model
from metrics.segmentation_metrics import dice_coef, iou_score
from utils.common import set_seed, get_transforms

class Trainer:
    def __init__(self, cfg:dict):
        self.cfg = cfg
        set_seed(cfg.get('seed',42))

        t_train, t_val = get_transforms(cfg['input_size'])
        self.train_ds = CocoSegmentationDataset(
            cfg['data']['train_images'], cfg['data']['train_ann'], t_train)
        self.val_ds = CocoSegmentationDataset(
            cfg['data']['val_images'], cfg['data']['val_ann'], t_val)

        self.train_loader = DataLoader(self.train_ds,
                                       batch_size=cfg['batch_size'],
                                       shuffle=True, num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(self.val_ds,
                                     batch_size=cfg['batch_size'],
                                     shuffle=False, num_workers=2, pin_memory=True)

        self.model, _ = get_model(cfg['model'], cfg['num_classes'])
        self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.opt = optim.AdamW(self.model.parameters(), lr=cfg['lr'])

        out = Path(cfg['output_dir'])
        out.mkdir(parents=True, exist_ok=True)
        self.best_path = out/'best.pt'

    def _run_loader(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        total_loss = total_dice = total_iou = 0.0
        for imgs, masks in tqdm(loader, leave=False):
            imgs, masks = imgs.cuda(), masks.cuda()
            logits = self.model(imgs)
            loss = self.criterion(logits, masks)
            if train:
                loss.backward(); self.opt.step(); self.opt.zero_grad()
            with torch.no_grad():
                preds = logits.argmax(1)
                total_loss += loss.item()
                total_dice += dice_coef(preds, masks).item()
                total_iou  += iou_score(preds, masks).item()
        n = len(loader)
        return total_loss/n, total_dice/n, total_iou/n

    def fit(self):
        best_iou = -1
        for epoch in range(self.cfg['epochs']):
            tr_loss, tr_dice, tr_iou = self._run_loader(self.train_loader, train=True)
            val_loss, val_dice, val_iou = self._run_loader(self.val_loader, train=False)
            print(f"[{epoch:02d}] tr IoU {tr_iou:.3f} val IoU {val_iou:.3f}")
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), self.best_path)
                print("New best model saved.")
