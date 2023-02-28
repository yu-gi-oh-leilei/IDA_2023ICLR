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
        feats = self.backbone(x)
        # feats = self.backbone(x)[0]
        
        if self.use_intervention:
            logits = self.clf(feats)
        else:
            logits = self.clf(feats.flatten(2).mean(-1))
        return logits, feats

 
class resnet101_backbone(Module):
    def __init__(self, pretrain=True):
        super(resnet101_backbone, self).__init__()

        if pretrain is True:
            WEIGHTDICT_V1 = {
                    'resnet18': 'ResNet18_Weights.IMAGENET1K_V1',
                    'resnet34': 'ResNet34_Weights.IMAGENET1K_V1',
                    'resnet50': 'ResNet50_Weights.IMAGENET1K_V1',
                    'resnet101': 'ResNet101_Weights.IMAGENET1K_V1',
                    }
            WEIGHTDICT_V2 = {
                    'resnet18': 'ResNet18_Weights.IMAGENET1K_V2',
                    'resnet34': 'ResNet34_Weights.IMAGENET1K_V2',
                    'resnet50': 'ResNet50_Weights.IMAGENET1K_V2',
                    'resnet101': 'ResNet101_Weights.IMAGENET1K_V2',
                    }
            # pretrained = WEIGHTDICT_V2['resnet101']
            pretrain = WEIGHTDICT_V1['resnet101']


        # res101 = torchvision.models.resnet101(pretrained=True)
        res101 = torchvision.models.resnet101(weights=pretrain)
        numFit = res101.fc.in_features
        self.resnet_layer = nn.Sequential(*list(res101.children())[:-2])
        
        self.num_channels = numFit

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
        self.num_channels = numFit
        del self.model.head

    def forward(self,x):
        feats = self.model.forward_features(x)
        return feats

def build_backbone(args):
    if args.backbone=="resnet101":
        backbone = resnet101_backbone(pretrain=True)
    elif args.backbone=="swim_transformer":
        backbone = swimtrans_backbone(num_classes=args.num_class, pretrain=True)
    elif args.backbone=="swim_transformer_large":
        backbone = swimtrans_backbone(num_classes=args.num_class, pretrain=True, large=True)
    return backbone

def build_ida(args):
    backbone = build_backbone(args)
    model = IDA(backbone=backbone, 
                num_classes=args.num_class, 
                use_intervention=args.use_intervention,
                num_head=args.nheads,
                heavy=args.heavy)

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    return model


























































