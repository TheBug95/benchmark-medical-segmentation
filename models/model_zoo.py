import timm, torch.nn as nn
from .sam_wrapper import build_sam_model

def get_model(name:str, num_classes:int):
    if name.startswith("sam") or name in ["mobilesam","medsam"]:
        return build_sam_model(name, num_classes)

    if name.startswith("vit"):
        from .vit_fcn import vit_fcn
        return vit_fcn(name, num_classes)

    if name.startswith("deeplab"):
        model = timm.create_model(name, pretrained=True, features_only=False, num_classes=num_classes)
        return model, 512

    if name.startswith("resnet"):
        backbone = timm.create_model(name, features_only=True, pretrained=True)
        ch = backbone.feature_info[-1]['num_chs']
        head = nn.Sequential(
            nn.Conv2d(ch, ch//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//2, num_classes, 1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        )
        model = nn.ModuleDict({'backbone': backbone, 'head': head})
        def forward(x):
            feats = model['backbone'](x)[-1]
            return model['head'](feats)
        model.forward = forward
        return model, 224

    raise ValueError(f"Unknown model '{name}'")
