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


def get_list_img(path):
    return os.listdir(path)

def label_prediction(model, img_path):
    img = Image.open(img_path).convert('RGB')
    img = preprocess_input(img).cuda()
    img = img.unsqueeze(0)
    with torch.no_grad():
        ans = model(img)
    return ans

def is_jeepsuv_label(label):
    return label[cfg_veh["class_map"]["jeep"]] >= 0.8 and label[cfg_veh["class_map"]["suv"]] >= 0.8

def predict_to_label(predict):
    ans = [int(x >= 0.6) for x in predict]
    return ans

def visualize_prediction(img_path, predict, config):
    CLASS_MAP = config["class_map"]
    text = ""
    for i in range(len(predict[0])):
        text += f'{CLASS_MAP[i]} {float(predict[0][i]):.2f}\n'
    
    row_dict = {}

    # for i, res in enumerate(predict[0]):
    #     if res >= 0.6:
    
    if is_jeepsuv_label(predict[0]):
        row_dict["paths"] = img_path
        row_dict["predict"] = predict[0].tolist()
        row_dict["label"] = predict_to_label(predict[0])
        
        img_copy = Image.open(img_path).convert('RGB')
        image_editable = ImageDraw.Draw(img_copy)
        image_editable.text((10,10), text)
        img_name = os.path.split(img_path)[-1]
        save_path = f"./results/classifier/{config['date']}/vehicle/visualize_test/jeep/{img_name}"
        img_copy.save(save_path)
    
    return row_dict


def visualize_data(model, folder, config):
    list_img = os.listdir(folder)
    data = []
    for img in tqdm(list_img):
        img_path = osp.join(folder, img)
        ans = label_prediction(model, img_path)

        row_dict = visualize_prediction(img_path, ans, config)
        if row_dict:
            row_dict["query_id"] = img_path
            data.append(row_dict)
    
    data_df = pd.DataFrame(data=data)
    return data_df
    


def main_veh():
    veh_model, _ = init_model(cfg_veh, cfg_col, load_ckpt=True, eval=True)
    veh_model.eval()
    veh_model = veh_model.cuda()

    root_dir = f"./results/classifier/{cfg_veh['date']}/vehicle/"
    save_dir = osp.join(root_dir, "visualize_test")
    os.makedirs(save_dir, exist_ok=True)
    
    for key, val in cfg_veh["class_map"].items():
        if type(val) == int:
            continue
        os.makedirs(osp.join(save_dir, val), exist_ok=True)
    
    folder = "/content/Track2/AIC21_Track2_ReID/image_test"
    data_df = visualize_data(veh_model, folder, cfg_veh)
    data_df.to_csv(osp.join(save_dir, "jeeplabel.csv"), index=False)


if __name__ == '__main__':
    main_veh()