from .registry import register
from .base import BaseStrategy
import torch, tqdm

@register("weakly")
class WeaklySupervisedStrategy(BaseStrategy):
    """
    Ejemplo mínimo: entrena con mapas de calor generados a partir de etiquetas débiles
    (por ejemplo bounding boxes). El trainer debe proporcionar 'weak_loader'.
    """
    def train_one_epoch(self, epoch:int):
        self.trainer.model.train()
        tot=0
        for imgs, heatmaps in tqdm.tqdm(self.trainer.weak_loader, leave=False):
            imgs, heatmaps = imgs.cuda(), heatmaps.cuda()
            logits = self.trainer.model(imgs)
            # Simple L2 entre logits softmax y heatmap objetivo
            loss = torch.mean((torch.softmax(logits,1)[:,1]-heatmaps)**2)
            loss.backward()
            self.trainer.opt.step(); self.trainer.opt.zero_grad()
            tot += loss.item()
        return tot/len(self.trainer.weak_loader)

    def validate(self, epoch:int):
        # Usa val_loader estándar con masks si las hay
        self.trainer.model.eval()
        tot=0
        with torch.no_grad():
            for imgs, masks in self.trainer.val_loader:
                imgs, masks = imgs.cuda(), masks.cuda()
                loss = self.trainer.criterion(self.trainer.model(imgs), masks)
                tot += loss.item()
        return tot/len(self.trainer.val_loader)
