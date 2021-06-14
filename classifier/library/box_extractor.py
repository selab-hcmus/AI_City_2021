import sys
# sys.path.append('./EfficientNet-PyTorch')
# from classifier.utils import preprocess_input
import numpy as np

import torch 
from torch import nn 
from torchvision import transforms

from classifier.models.efficientnet_pytorch import EfficientNet
from classifier.utils import preprocess_input

class BoxClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = EfficientNet.from_pretrained(
            cfg['MODEL'], num_classes=cfg['NUM_CLASSES'], include_top=False)
        out_channel = backbone._conv_head.out_channels

        self.feature_extractor = nn.Sequential(
            backbone,
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(out_channel, cfg['NUM_CLASSES'])
        )

        self.logit_activation = nn.Identity()
        if cfg['output_type'] == 'one_hot':
            self.logit_activation = nn.Sigmoid()
        elif cfg['output_type'] == 'fraction':
            self.logit_activation = nn.Softmax(dim=-1)

    def extract_feature(self, input):
        x = self.feature_extractor(input)
        # x = self.avg_pool(x)
        return x
        
    def forward(self, input):
        x = self.extract_feature(input)
        logits = self.classifier(x)
        logits = self.logit_activation(logits)

        return logits

    def predict_given_feat(self, box_feat):
        logits = self.classifier(box_feat)
        logits = self.logit_activation(logits)

        return logits

    def predict(self, images: list):
        list_boxes = [preprocess_input(img) for img in images]
        inp = torch.stack(list_boxes, dim=0).cuda()
        preds = self.forward(inp).detach().cpu().numpy()
        return preds
    pass

class VehicleClassifier(BoxClassifier):
    def __init__(self, cfg) -> None: 
        super().__init__(cfg)
        
class ColorClassifier(BoxClassifier):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

def get_state_dict(weight_path):

    state_dict = torch.load(weight_path)
    if state_dict.get("feature_extractor.0._fc.weight") is not None:
        state_dict.pop("feature_extractor.0._fc.weight")
    if state_dict.get("feature_extractor.0._fc.bias") is not None:
        state_dict.pop("feature_extractor.0._fc.bias")
    return state_dict


def init_model(cfg_veh, cfg_col, load_ckpt=True, eval=False):
    veh_model = VehicleClassifier(cfg_veh)
    col_model = ColorClassifier(cfg_col)

    if eval:
        veh_model.eval()
        col_model.eval()

    if load_ckpt:
        veh_weight = cfg_veh['WEIGHT']
        col_weight = cfg_col['WEIGHT']
        print(f'load veh weight from {veh_weight}')
        print(f'load col weight from {col_weight}')

        veh_model.load_state_dict(get_state_dict(veh_weight)) 
        col_model.load_state_dict(get_state_dict(col_weight))    

    return veh_model, col_model
