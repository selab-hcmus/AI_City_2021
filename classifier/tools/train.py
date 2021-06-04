import os, json, cv2
import os.path as osp 
import pandas as pd 
from tqdm.notebook import tqdm
from glob import glob
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, Dataset

from classifier.box_extractor import init_model
from classifier.config import cfg_veh, cfg_col
from classifier.dataset import (
    VehicleDataset, get_dataset,
    VEH_TRAIN_CSV, COL_TRAIN_CSV,
    VEH_GROUP_JSON, COL_GROUP_JSON,
    VEH_BOX_DIR, COL_BOX_DIR,
) 
import classifier.loss
from classifier.utils import scan_data, evaluate_fraction, evaluate_tensor, train_model

UP_TRAIN = cfg_veh['uptrain']
if UP_TRAIN:
    veh_model, col_model = init_model(cfg_veh, cfg_col, load_ckpt=True, eval=False)
else:
    veh_model, col_model = init_model(cfg_veh, cfg_col, load_ckpt=False, eval=False)
veh_model = veh_model.cuda()
col_model = col_model.cuda()


def train_model_type(model, cfg, csv_path: str, json_path: str, box_dir: str, up_train:bool=False):
    df_train, df_val, df_full = get_dataset(csv_path, json_path, box_dir)
    if up_train:
        df_train = df_full

    train_dataset = VehicleDataset(df_train, 'train')
    val_dataset = VehicleDataset(df_val, 'val')

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['val']['batch_size'], shuffle=False, num_workers=2)
    
    # test 
    print(f"train dataset: {len(train_dataset)}")
    print(f"val dataset: {len(val_dataset)}")
    sample = train_dataset[0]
    for k in sample.keys():
        print(f'{k} shape: {sample[k].shape}')

    scan_data(train_dataloader, name='Train')
    scan_data(val_dataloader, name='Val')

    # criterion = l2_loss()
    criterion_class = getattr(loss, cfg_veh['loss']['name'])
    criterion = criterion_class(**cfg_veh['loss']['args'])
    # criterion = BceDiceLoss(weight_bce=0.0, weight_dice=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, min_lr=1e-07, eps=1e-07, verbose=True)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader

    save_path = osp.join(cfg['save_path'], cfg['date'], cfg['type'])
    os.makedirs(save_path, exist_ok=True)
    print(f'Save model to {save_path}')
    
    df_train.to_csv(osp.join(save_path, "train_df.csv"), index = False)
    df_val.to_csv(osp.join(save_path, "val_df.csv"), index = False)

    model, val_acc, train_acc = train_model(
        model, dataloaders, 
        criterion, optimizer, lr_scheduler, 
        num_epochs=cfg['train']['num_epochs'], 
        save_path=save_path
    )
    pass

def train_vehicle():
    print(f'TRAIN VEHICLE')
    train_model_type(veh_model, cfg_veh, VEH_TRAIN_CSV, VEH_GROUP_JSON, VEH_BOX_DIR, UP_TRAIN)
    pass

def train_color():
    print(f'TRAIN COLOR')
    train_model_type(col_model, cfg_col, COL_TRAIN_CSV, COL_GROUP_JSON, COL_BOX_DIR, False)
    pass


import seaborn as sns
def test_loss():
    # test lr
    criterion_class = getattr(loss, cfg_veh['loss']['name'])
    criterion = criterion_class(**cfg_veh['loss']['args'])
    list_thres = []
    list_it = []
    print(cfg_veh['loss'])
    for i in range(criterion.num_steps):
        list_it.append(i)
        list_thres.append(criterion.threshold().item())
        criterion.step()
        pass
    plot = sns.lineplot(x=list_it, y=list_thres)
    plot.figure.savefig('./tsa_thres_alpha3.png')
    pass


def main():
    train_vehicle()
    # train_color()
    pass

if __name__ == '__main__':
    main()
