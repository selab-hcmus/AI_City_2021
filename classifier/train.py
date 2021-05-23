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
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader, Dataset

from box_extractor import init_model
from config import cfg_veh, cfg_col
from dataset import (
    VehicleDataset, get_dataset,
    VEH_TRAIN_CSV, COL_TRAIN_CSV,
    VEH_GROUP_JSON, COL_GROUP_JSON,
    VEH_BOX_DIR, COL_BOX_DIR,
) 
from utils import (
    l2_loss, evaluate_fraction, evaluate_tensor, train_model
)

import torch, gc
gc.collect()
torch.cuda.empty_cache()


veh_model, col_model = init_model(cfg_veh, cfg_col, load_ckpt=False)
veh_model = veh_model.cuda()
col_model = col_model.cuda()

def train_model_type(model, cfg, csv_path: str, json_path: str, box_dir: str):
    df_train, df_val = get_dataset(csv_path, json_path, box_dir)

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

    criterion = l2_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-07, eps=1e-07, verbose=True)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader

    save_path = osp.join(cfg['save_path'], cfg['date'], cfg['type'])
    os.makedirs(save_path, exist_ok=True)
    print("Created save directory")

    df_train.to_csv(osp.join(save_path, "train_df.csv"), index = False)
    df_val.to_csv(osp.join(save_path, "val_df.csv"), index = False)

    model, val_acc, train_acc = train_model(
        model, dataloaders, 
        criterion, optimizer, lr_scheduler, 
        num_epochs=cfg['train']['num_epochs'], 
        save_path=osp.join(save_path)
    )
    pass

def train_vehicle():
    print(f'TRAIN VEHICLE')
    train_model_type(veh_model, cfg_veh, VEH_TRAIN_CSV, VEH_GROUP_JSON, VEH_BOX_DIR)
    pass

def train_color():
    print(f'TRAIN COLOR')
    train_model_type(col_model, cfg_col, COL_TRAIN_CSV, COL_GROUP_JSON, COL_BOX_DIR)
    pass


def main():
    train_vehicle()
    # train_color()
    pass

if __name__ == '__main__':
    main()
    pass