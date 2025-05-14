import torch, json
from torch.utils.data import DataLoader
from datasets.coco_segmentation import CocoSegmentationDataset
from models.model_zoo import get_model
from metrics.segmentation_metrics import dice_coef, iou_score
from utils.common import get_transforms

def test(cfg, weights):
    _, t_val = get_transforms(cfg['input_size'])
    ds = CocoSegmentationDataset(cfg['data']['test_images'],
                                 cfg['data']['test_ann'], t_val)
    loader = DataLoader(ds, batch_size=cfg['batch_size'],
                        shuffle=False, num_workers=2, pin_memory=True)
    model, _ = get_model(cfg['model'], cfg['num_classes'])
    sd = torch.load(weights, map_location='cpu')
    model.load_state_dict(sd)
    model.cuda(); model.eval()

    tot_dice = tot_iou = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.cuda(), masks.cuda()
            preds = model(imgs).argmax(1)
            tot_dice += dice_coef(preds, masks).item()
            tot_iou  += iou_score(preds, masks).item()
    n = len(loader)
    print(f"TEST | Dice {tot_dice/n:.3f} IoU {tot_iou/n:.3f}")
