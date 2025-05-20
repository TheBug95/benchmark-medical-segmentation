from .registry import register
from .base import BaseStrategy
import torch, tqdm

@register("finetune")
class FinetuneStrategy(BaseStrategy):
    """
    Fase 1: congela el backbone durante N epochs.
    Fase 2: entrenamiento fine-tune completo.
    """
    def __init__(self, trainer, freeze_epochs:int=5):
        super().__init__(trainer)
        self.freeze_epochs = freeze_epochs
        # freeze backbone
        for n,p in trainer.model.named_parameters():
            if "backbone" in n:
                p.requires_grad = False

    def train_one_epoch(self, epoch:int):
        if epoch == self.freeze_epochs:
            for p in self.trainer.model.parameters():
                p.requires_grad = True
        self.trainer.model.train()
        tot=0
        for imgs, masks in tqdm.tqdm(self.trainer.train_loader, leave=False):
            imgs, masks = imgs.cuda(), masks.cuda()
            logits = self.trainer.model(imgs)
            loss = self.trainer.criterion(logits, masks)
            loss.backward()
            self.trainer.opt.step(); self.trainer.opt.zero_grad()
            tot += loss.item()
        return tot/len(self.trainer.train_loader)

    def validate(self, epoch:int):
        self.trainer.model.eval()
        tot=0
        with torch.no_grad():
            for imgs, masks in self.trainer.val_loader:
                imgs, masks = imgs.cuda(), masks.cuda()
                loss = self.trainer.criterion(self.trainer.model(imgs), masks)
                tot += loss.item()
        return tot/len(self.trainer.val_loader)
