import datetime
import os
import os.path as osp 

VEH_CLASS_MAP = {
    0: 'sedan',
    1: 'suv',
    2: 'van',
    3: 'jeep',
    4: 'pickup',
    5: 'bus-truck',
}
COL_CLASS_MAP = {
    0: 'gray',
    1: 'gold',
    2: 'red',
    3: 'blue',
    4: 'black',
    5: 'purple',
    6: 'green',
    7: 'white',
}

CONFIG = {
    'MODEL': 'efficientnet-b5',
    'NUM_CLASSES': 0,
    'image_size': (224,224),
    "imagenet_mean":[0.485, 0.456, 0.406],
    "imagenet_std":[0.229, 0.224, 0.225],
    'score_thres': 0.5,
    'seed': 88,
    'uptrain': True,
    
    'train': {
        'batch_size': 32,
        'num_epochs': 10,
    },
    'val':{
        'batch_size': 32,
    },
    'date': "May31_uptrain", 
    'save_path': "./results/classifier", #Training log will be saved at save_path/<mode>/date
    # 'loss': {
    #     'name': 'TSA_BceDiceLoss',
    #     'args': {
    #         'weight_bce': 0.0,
    #         'weight_dice': 1.0,
    #         'alpha': 5.0, 
    #         'num_classes': None,
    #         'num_steps': None
    #     }
    # },
    'loss': {
        'name': 'BceDiceLoss',
        'args': {
            'weight_bce': 0.0,
            'weight_dice': 1.0
        }
    },
    # 'loss': {
    #     'name': 'l2_loss',
    #     'args': {
    #         'reduction': 'mean'
    #     }
    # }
}

cfg_veh = CONFIG.copy()
cfg_col = CONFIG.copy()

cfg_veh.update({
    'NUM_CLASSES': 6,
    'WEIGHT': f"/home/ntphat/projects/AI_City_2021/classifier/results/May31_uptrain/veh_best_model.pt",
    "type": "vehicle",
    'output_type': 'one_hot',
    'class_map': {
        0: 'sedan',
        1: 'suv',
        2: 'van',
        3: 'jeep',
        4: 'pickup',
        5: 'bus-truck',
        'sedan': 0,
        'suv': 1,
        'van': 2,
        'jeep': 3,
        'pickup': 4,
        'bus-truck': 5
    },
})

cfg_col.update({
    'NUM_CLASSES': 8,
    'WEIGHT': "/home/ntphat/projects/AI_City_2021/classifier/results/col_classifier.pt",
    "type": "color",
    'output_type': 'one_hot',
    'class_map': {
        0: 'gray',
        1: 'gold',
        2: 'red',
        3: 'blue',
        4: 'black',
        5: 'purple',
        6: 'green',
        7: 'white',
    }
})

def setup_cfg(cfg):
    cfg['loss']['args']['num_steps'] = cfg['train']['batch_size']*cfg['train']['num_epochs']
    cfg['loss']['args']['num_classes'] = cfg['NUM_CLASSES']
    pass

if CONFIG['loss']['name'] == 'TSA_BceDiceLoss':
    setup_cfg(cfg_veh)
    setup_cfg(cfg_col)
