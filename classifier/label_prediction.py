from box_extractor import init_model, preprocess_input
from config import cfg_veh, cfg_col
from tqdm import tqdm
import json
from PIL import Image
import torch
import os

TRACKS_DIR = '../dataset/data/test-tracks.json'
track = json.load(open(TRACKS_DIR))

def label_extraction(model, config, track):
    ans = {}
    split = 10
    for key in tqdm(track):
        l = len(track[key]["boxes"])
        idx = split_data(range(l), min(split,l))
        frames = [track[key]['frames'][i] for i in idx]
        boxes = [track[key]['boxes'][i] for i in idx]
        pred = []
        for frame, box in zip(frames, boxes):
            new_path = os.path.join("/content/test_imgs", frame)
            img = Image.open(new_path)
            x, y, w, h = box
            x_0, y_0, x_1, y_1 = int(x), int(y), int(x+w), int(y+w)
            crop_img = img.crop((x_0, y_0, x_1, y_1))
            crop_img = preprocess_input(crop_img).cuda()
            crop_img = crop_img.unsqueeze(0)
            with torch.no_grad():
                pred.append(model(crop_img))
        pred = torch.mean(torch.stack(pred), dim=0).cpu().detach().numpy().tolist() 
        ans[key] = pred
    return ans

def split_data(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    ans = []
    for arr in out:
        ans.append(arr[len(arr)//2])
    return ans





if __name__ == "__main__":
    veh_model, col_model = init_model(cfg_veh, cfg_col, load_ckpt=True)

    veh_model = veh_model.cuda()
    col_model = col_model.cuda()

    veh_model.eval()
    col_model.eval()

    # predict vehicle label for test track's visual data
    label_extraction(veh_model, cfg_veh, track)

    # predict vehicle label for test track's visual data
    label_extraction(col_model, cfg_col, track)
