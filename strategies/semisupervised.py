from .registry import register
from .base import BaseStrategy
import torch, tqdm, copy

@register("semisupervised")
class MeanTeacherStrategy(BaseStrategy):
    """
    Implementa esquema Mean Teacher simplificado (consistency loss).
    Requiere que el trainer disponga de train_loader_ul (unlabeled).
    """
    def __init__(self, trainer, ema_decay:float=0.99, consistency_w:float=1.0):
        super().__init__(trainer)
        self.ema_model = copy.deepcopy(trainer.model)
        for p in self.ema_model.parameters(): p.requires_grad = False
        self.decay = ema_decay
        self.cons_w = consistency_w

    def _update_ema(self):
        for ema_p, p in zip(self.ema_model.parameters(), self.trainer.model.parameters()):
            ema_p.data = self.decay*ema_p.data + (1-self.decay)*p.data

    def train_one_epoch(self, epoch:int):
        self.trainer.model.train()
        tot=0
        for ((img_l, mask_l), (img_u,_)) in tqdm.tqdm(zip(self.trainer.train_loader,
                                                          self.trainer.train_loader_ul),
                                                      total=len(self.trainer.train_loader), leave=False):
            img_l, mask_l = img_l.cuda(), mask_l.cuda()
            img_u = img_u.cuda()
            logits_l = self.trainer.model(img_l)
            sup_loss = self.trainer.criterion(logits_l, mask_l)
            # Consistency on unlabeled
            with torch.no_grad():
                teacher_logits = self.ema_model(img_u)
            stud_logits = self.trainer.model(img_u)
            cons_loss = torch.mean((stud_logits - teacher_logits.detach())**2)
            loss = sup_loss + self.cons_w*cons_loss
            loss.backward()
            self.trainer.opt.step(); self.trainer.opt.zero_grad()
            self._update_ema()
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
