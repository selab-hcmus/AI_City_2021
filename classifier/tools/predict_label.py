import classifier
from tqdm import tqdm
import json
from PIL import Image
import torch
import os
import os.path as osp 

from classifier.library.box_extractor import init_model
from classifier.library.manager import ClassifierManager

from classifier.utils import preprocess_input
from classifier.utils.config import cfg_veh, cfg_col
from utils import RESULT_DIR, TEST_TRACK_JSON, DATA_DIR, test_track_map, dict_save, prepare_dir


TRACKS_DIR = TEST_TRACK_JSON #'../dataset/data/test-tracks.json'
track = json.load(open(TRACKS_DIR))
save_dir = prepare_dir(osp.join(RESULT_DIR, 'classifier', "predict_label")) 

class_manager = ClassifierManager()

def label_extraction(model, track):
    ans = {}
    split = 10
    for key in tqdm(track):
        
        new_id = test_track_map[key]
        if new_id not in [3, '3']:
            continue 
        
        print(key)
        # print(track[key]['frames'])


        l = len(track[key]["boxes"])
        
        idx = split_data(range(l), min(split,l))
        frames = [track[key]['frames'][i] for i in idx]
        boxes = [track[key]['boxes'][i] for i in idx]
        
        pred = []
        for frame, box in zip(frames, boxes):
            print(frame)
            new_path = osp.join(DATA_DIR, frame)
            
            img = Image.open(new_path)
            x, y, w, h = box
            x_0, y_0, x_1, y_1 = int(x), int(y), int(x+w), int(y+h)
            print((x_0, y_0, x_1, y_1))
            crop_img = img.crop((x_0, y_0, x_1, y_1))
            crop_img = preprocess_input(crop_img).cuda()
            crop_img = crop_img.unsqueeze(0)

            with torch.no_grad():
                # pred.append(model(crop_img))
                pred.append(class_manager.col_model(crop_img))

        pred = torch.stack(pred)
        print(pred)
        pred = torch.mean(pred, dim=0).cpu().detach().numpy().tolist() 
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

def save_result(save_dir, data_json):
    f = open(save_dir, 'w')
    json.dump(data_json, f, indent=2)
    f.close()

def main2():
    

    pass

if __name__ == "__main__":
    veh_model, col_model = init_model(cfg_veh, cfg_col, load_ckpt=True, eval=True)

    veh_model = veh_model.cuda()
    col_model = col_model.cuda()

    # # predict vehicle label for test track's visual data
    # print('Vehicle prediction')

    # ans_veh = label_extraction(veh_model, track)
    # save_dir_veh = os.path.join(save_dir, "test_vehicle_predict.json")
    # dict_save(ans_veh, save_dir_veh)

    # predict color label for test track's visual data
    print('Color prediction')

    col_veh = label_extraction(col_model, track)
    save_dir_col = os.path.join(save_dir, "test_color_predict.json")
    dict_save(col_veh, save_dir_col)
