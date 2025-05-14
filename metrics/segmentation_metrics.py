import torch

def _intersection_union(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = torch.sum((pred == target) & (target > 0)).float()
    union = torch.sum(pred > 0).float() + torch.sum(target > 0).float() - inter
    return inter, union

def dice_coef(pred, target, eps=1e-6):
    inter = 2 * torch.sum((pred == target) & (target > 0)).float()
    denom = torch.sum(pred > 0).float() + torch.sum(target > 0).float()
    return (inter + eps) / (denom + eps)

def iou_score(pred, target, eps=1e-6):
    inter, union = _intersection_union(pred, target)
    return (inter + eps) / (union + eps)
