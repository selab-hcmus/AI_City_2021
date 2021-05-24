from tqdm import tqdm
import torch
import os
import os.path as osp 
from PIL import Image, ImageFont, ImageDraw 
from config import cfg_veh, cfg_col
import pandas as pd

from box_extractor import init_model
from utils import preprocess_input
from config import cfg_veh, cfg_col
from ast import literal_eval

def label_prediction(model, img_path):
    img = Image.open(img_path).convert('RGB')
    img = preprocess_input(img).cuda()
    img = img.unsqueeze(0)
    with torch.no_grad():
        ans = model(img)
    return ans

def visualize_prediction(img_path, label, predict, config):
    CLASS_MAP = config["class_map"]
    text = ""
    for i in range(len(predict[0])):
        text += f'{CLASS_MAP[i]} {float(label[i]):.1f} {float(predict[0][i]):.2f}\n'
    
    for i, res in enumerate(predict[0]):
        if res >= 0.6:
            img_copy = Image.open(img_path).convert('RGB')
            image_editable = ImageDraw.Draw(img_copy)
            image_editable.text((10,10), text)
            img_name = os.path.split(img_path)[-1]
            save_path = f"./results/classifier/{config['date']}/vehicle/visualize/{CLASS_MAP[i]}/{img_name}"
            img_copy.save(save_path)

def visualize_data(model, data_path, config):
    data_df = pd.read_csv(data_path)

    for index, row in tqdm(data_df.iterrows()):
        ans = label_prediction(model, row["paths"])
        visualize_prediction(row["paths"], literal_eval(row["labels"]), ans, config)

def main_veh():
    veh_model, _ = init_model(cfg_veh, cfg_col, load_ckpt=True, eval=True)
    veh_model = veh_model.cuda()

    root_dir = f"./results/classifier/{cfg_veh['date']}/vehicle/"
    save_dir = osp.join(root_dir, "visualize")
    os.makedirs(save_dir, exist_ok=True)
    
    for key, val in cfg_veh["class_map"].items():
        os.makedirs(osp.join(save_dir, val), exist_ok=True)
    
    data_path_csv = osp.join(root_dir, "val_df.csv")
    visualize_data(veh_model, data_path_csv, cfg_veh)

if __name__ == '__main__':
    main_veh()