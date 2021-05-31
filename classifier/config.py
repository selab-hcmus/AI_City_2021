import datetime
import os
import os.path as osp 

CONFIG = {
    'MODEL': 'efficientnet-b5',
    'NUM_CLASSES': 0,
    'image_size': (224,224),
    "imagenet_mean":[0.485, 0.456, 0.406],
    "imagenet_std":[0.229, 0.224, 0.225],
    'score_thres': 0.5,
    'seed': 88,

    'train': {
        'batch_size': 32,
        'num_epochs': 50,
    },
    'val':{
        'batch_size': 32,
    },
    'save_path': "./results/classifier",
}
cfg_veh = CONFIG.copy()
cfg_col = CONFIG.copy()

date ="May31"

cfg_veh.update({
    'NUM_CLASSES': 6,
    'WEIGHT': f"./results/classifier/May30/vehicle/last_model.pt",
    "date": date,
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
    'WEIGHT': "./results/col_classifier.pt",
    "date": date,
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