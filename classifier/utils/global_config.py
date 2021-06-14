import os 
import os.path as osp

NUM_CLASSES = {
    'street': 3, 'cam': 31
}

CONFIG = {
    'MODEL': 'efficientnet-b5',
    'NUM_CLASSES': 0,
    'image_size': (224,224),
    "imagenet_mean":[0.485, 0.456, 0.406],
    "imagenet_std":[0.229, 0.224, 0.225],
    'score_thres': 0.5,
    'seed': 88,
    'uptrain': False,
    
    'train': {
        'batch_size': 32,
        'num_epochs': 10,
    },
    'val':{
        'batch_size': 32,
    },
    'date': "May31_uptrain", 
    'save_path': "./results/", #Training log will be saved at save_path/<mode>/date
}

def setup_cfg(cfg: dict, mode: str='street'):
    
    pass
