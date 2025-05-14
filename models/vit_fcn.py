import timm, torch.nn as nn

def vit_fcn(name:str, num_classes:int):
    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, features_only=True)
    ch = backbone.feature_info[-1]['num_chs']
    head = nn.Sequential(
        nn.Conv2d(ch, ch//2, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch//2, num_classes, 1),
        nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    )
    model = nn.ModuleDict({'backbone': backbone, 'head': head})
    def forward(x):
        feats = model['backbone'](x)[-1]
        logits = model['head'](feats)
        return logits
    model.forward = forward
    return model, 224
