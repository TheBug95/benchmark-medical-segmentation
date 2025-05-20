import torch
from torch.utils.data import DataLoader
from datasets.coco_segmentation import CocoSegmentationDataset
from models.model_zoo import get_model
from metrics.segmentation_metrics import dice_coef, iou_score
from utils.common import get_transforms

def test(cfg, weights):
    _, tval = get_transforms(cfg['input_size'])
    ds = CocoSegmentationDataset(cfg['data']['test_images'], cfg['data']['test_ann'], tval)
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    model,_ = get_model(cfg['model'], cfg['num_classes'])
    model.load_state_dict(torch.load(weights, map_location='cpu'))
    model.cuda(); model.eval()
    dice_sum=iou_sum=0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.cuda(), masks.cuda()
            preds = model(imgs).argmax(1)
            dice_sum += dice_coef(preds, masks).item()
            iou_sum  += iou_score(preds, masks).item()
    n = len(loader)
    print(f"TEST | Dice {dice_sum/n:.3f} IoU {iou_sum/n:.3f}")
