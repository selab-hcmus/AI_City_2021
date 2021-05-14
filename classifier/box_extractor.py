import sys
sys.path.append('./EfficientNet-PyTorch')
from utils import preprocess_input
import torch 
from torch import nn 
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
import PIL
IMAGE_SIZE = (224,224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class VehicleClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = EfficientNet.from_pretrained(cfg['MODEL'], num_classes=cfg['NUM_CLASSES'], include_top=False)
        # backbone._fc = nn.Identity()
        # backbone._dropout = nn.Identity()
        # self.backbone = self.backbone.cuda()
        out_channel = backbone._conv_head.out_channels

        self.feature_extractor = nn.Sequential(
            backbone,
            nn.Flatten()
        )
        # avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(out_channel, cfg['NUM_CLASSES'])
        )
        # self.classifier = nn.Linear(out_channel, cfg['NUM_CLASSES'])
        
    def extract_feature(self, input):
        x = self.feature_extractor(input)
        # x = self.avg_pool(x)
        return x
        
    def forward(self, input):
        x = self.extract_feature(input)
        logits = self.classifier(x)
        # logits = self.skeleton(input)
        logits = torch.softmax(logits, dim=-1)
        return logits

class ColorClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = EfficientNet.from_pretrained(cfg['MODEL'], num_classes=cfg['NUM_CLASSES'], include_top=False)
        # self.backbone = self.backbone.cuda()
        out_channel = backbone._conv_head.out_channels

        self.feature_extractor = nn.Sequential(
            backbone,
            nn.Flatten()
        )
        # avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(out_channel, cfg['NUM_CLASSES'])
        )
        # self.classifier = nn.Linear(out_channel, cfg['NUM_CLASSES'])
        
    def extract_feature(self, input):
        x = self.feature_extractor(input)
        # x = self.avg_pool(x)
        return x
        
    def forward(self, input):
        x = self.extract_feature(input)
        logits = self.classifier(x)
        # logits = self.skeleton(input)
        logits = torch.softmax(logits, dim=-1)
        return logits

def get_state_dict(weight_path):
    state_dict = torch.load(weight_path)
    state_dict.pop("feature_extractor.0._fc.weight")
    state_dict.pop("feature_extractor.0._fc.bias")
    return state_dict
    pass

def init_model(cfg_veh, cfg_col, load_ckpt=True):
    veh_model = VehicleClassifier(cfg_veh)
    col_model = ColorClassifier(cfg_col)

    # veh_model.eval()
    # col_model.eval()

    if load_ckpt:
        veh_weight = cfg_veh['WEIGHT']
        col_weight = cfg_col['WEIGHT']

        veh_model.load_state_dict(get_state_dict(veh_weight)) 
        col_model.load_state_dict(get_state_dict(col_weight))    

    return veh_model, col_model
