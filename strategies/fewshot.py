from .registry import register
from .base import BaseStrategy
import torch, random, tqdm
from torch.utils.data import DataLoader, Subset

@register("fewshot")
class FewShotStrategy(BaseStrategy):
    """
    Entrenamiento episodico similar a Prototypical Networks (super sencillo).
    cfg debe contener 'k_shot' y 'n_episodes'.
    """
    def __init__(self, trainer, k_shot:int=5, n_episodes:int=100):
        super().__init__(trainer)
        self.k = k_shot
        self.n_eps = n_episodes

    def _sample_episode(self):
        # muestrea k ejemplos por clase del dataset de entrenamiento
        cls_to_idxs = {}
        for idx,(_,mask) in enumerate(self.trainer.train_ds):
            cls = 1 if mask.sum()>0 else 0
            cls_to_idxs.setdefault(cls, []).append(idx)
        support_idxs=[]; query_idxs=[]
        for c,idxs in cls_to_idxs.items():
            random.shuffle(idxs)
            support_idxs+=idxs[:self.k]
            query_idxs+=idxs[self.k:self.k+5]
        sup_loader=DataLoader(Subset(self.trainer.train_ds,support_idxs),
                              batch_size=len(support_idxs))
        qry_loader=DataLoader(Subset(self.trainer.train_ds,query_idxs),
                              batch_size=len(query_idxs))
        return next(iter(sup_loader)), next(iter(qry_loader))

    def train_one_epoch(self, epoch:int):
        self.trainer.model.train()
        tot=0
        for _ in tqdm.tqdm(range(self.n_eps), leave=False):
            (sup_imgs, sup_masks), (qry_imgs, qry_masks) = self._sample_episode()
            sup_imgs, sup_masks = sup_imgs.cuda(), sup_masks.cuda()
            qry_imgs, qry_masks = qry_imgs.cuda(), qry_masks.cuda()
            # Simple fine-tune on support then evaluate on query
            logits = self.trainer.model(sup_imgs)
            loss = self.trainer.criterion(logits, sup_masks)
            loss.backward()
            self.trainer.opt.step(); self.trainer.opt.zero_grad()
            # evaluate query loss for logging
            with torch.no_grad():
                q_loss = self.trainer.criterion(self.trainer.model(qry_imgs), qry_masks)
            tot += q_loss.item()
        return tot/self.n_eps

    def validate(self, epoch:int):
        # usa validación estándar
        self.trainer.model.eval()
        tot=0
        with torch.no_grad():
            for imgs, masks in self.trainer.val_loader:
                imgs, masks = imgs.cuda(), masks.cuda()
                loss = self.trainer.criterion(self.trainer.model(imgs), masks)
                tot += loss.item()
        return tot/len(self.trainer.val_loader)
