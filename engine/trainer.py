import torch, numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datasets.coco_segmentation import CocoSegmentationDataset
from models.model_zoo import get_model
from metrics.segmentation_metrics import dice_coef, iou_score
from utils.common import set_seed, get_transforms
from strategies import STRATEGY_REGISTRY

class Trainer:
    def __init__(self, cfg:dict):
        self.cfg = cfg
        set_seed(cfg.get('seed',42))
        t_train, t_val = get_transforms(cfg['input_size'])
        self.train_ds = CocoSegmentationDataset(cfg['data']['train_images'],
                                                cfg['data']['train_ann'], t_train)
        self.val_ds = CocoSegmentationDataset(cfg['data']['val_images'],
                                              cfg['data']['val_ann'], t_val)
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

        # instantiate strategy
        strat_name = cfg.get('strategy', 'standard')
        if strat_name == 'standard':
            self.strategy = None
        else:
            StrategyClass = STRATEGY_REGISTRY.get(strat_name)
            assert StrategyClass, f"Estrategia '{strat_name}' no registrada"
            self.strategy = StrategyClass(self, **cfg.get('strategy_kwargs', {}))

        out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
        self.best_path = out/'best.pt'

    def _standard_epoch(self, train=True):
        loader = self.train_loader if train else self.val_loader
        self.model.train() if train else self.model.eval()
        tot=0
        for imgs, masks in tqdm(loader, leave=False):
            imgs, masks = imgs.cuda(), masks.cuda()
            logits = self.model(imgs)
            loss = self.criterion(logits, masks)
            if train:
                loss.backward(); self.opt.step(); self.opt.zero_grad()
            tot+=loss.item()
        return tot/len(loader)

    def fit(self):
        best = 1e9
        for epoch in range(self.cfg['epochs']):
            if self.strategy:
                tr_loss = self.strategy.train_one_epoch(epoch)
                val_loss = self.strategy.validate(epoch)
            else:
                tr_loss = self._standard_epoch(train=True)
                val_loss = self._standard_epoch(train=False)
            print(f"[{epoch:02d}] tr {tr_loss:.3f} val {val_loss:.3f}")
            if val_loss < best:
                best=val_loss
                torch.save(self.model.state_dict(), self.best_path)
                print("New best model saved at", self.best_path)
