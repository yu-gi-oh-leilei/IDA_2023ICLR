import torch
import torch.nn as nn
from torch.nn import Module as Module
from collections import OrderedDict
import torchvision
import math
import numpy as np
from models.backbone_wope import build_backbone_wope
from models.classifier import Interventional_Classifier, CosNorm_Classifier, attention_layer, attention_layers, attention 
from models.vision_transformer.swin_transformer import SwinTransformer


class IDA(Module):
    def __init__(self, backbone, num_classes=80, use_intervention=True, num_head=4, heavy=True):
        super(IDA,self).__init__()

        self.backbone = backbone
        

        self.feat_dim = self.backbone.num_channels
        self.use_intervention = use_intervention
        
        if not use_intervention:
            print('use linear')
            self.clf = nn.Linear(self.feat_dim, num_classes)
        else:
            print('use intervention')
            self.clf = Interventional_Classifier(num_classes=num_classes, 
                                                 feat_dim=self.feat_dim, 
                                                 num_head=num_head, 
                                                 beta=0.03125, 
                                                 heavy=heavy)

    def forward(self,x):
        # feats = self.backbone(x)
        feats = self.backbone(x)[0]
        
        if self.use_intervention:
            logits = self.clf(feats)
        else:
            logits = self.clf(feats.flatten(2).mean(-1))
        return logits, feats

 
class resnet101_backbone(Module):
    def __init__(self, pretrain):
        super(resnet101_backbone,self).__init__()
        res101 = torchvision.models.resnet101(pretrained=True)
        numFit = res101.fc.in_features
        self.resnet_layer = nn.Sequential(*list(res101.children())[:-2])
        
        self.feat_dim = numFit

    def forward(self,x):
        feats = self.resnet_layer(x)
        
        return feats

class swimtrans_backbone(Module):
    def __init__(self, num_classes, pretrain, large=False):
        super(swimtrans_backbone,self).__init__()
        if large:
            self.model = SwinTransformer(img_size=384,patch_size=4,num_classes=num_classes,embed_dim=192,depths=(2, 2, 18, 2),num_heads=(6, 12, 24, 48),window_size=12)
        else:
            self.model = SwinTransformer(img_size=384,patch_size=4,num_classes=num_classes,embed_dim=128,depths=(2, 2, 18, 2),num_heads=(4, 8, 16, 32),window_size=12)
        if pretrain:
            path = pretrain
            state = torch.load(path, map_location='cpu')['model']
            filtered_dict = {k: v for k, v in state.items() if(k in self.model.state_dict() and 'head' not in k)}
            self.model.load_state_dict(filtered_dict,strict=False)
        numFit = self.model.num_features
        self.feat_dim = numFit
        del self.model.head

    def forward(self,x):
        feats = self.model.forward_features(x)
        return feats

def build_ida(args):
    backbone = build_backbone_wope(args)
    model = IDA(backbone=backbone, 
                num_classes=args.num_class, 
                use_intervention=args.use_intervention,
                num_head=args.nheads,
                heavy=args.heavy)

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    return model


























































