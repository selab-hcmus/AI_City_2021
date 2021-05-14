import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2,pickle,sys
import os
from deepsort import *
import json
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

COLAB = True

TRACK_DIR = "/content/SOURCE/vehicle_tracking/data/Centernet2_train_veh_boxes.json"
ROOT_DIR = "/content/DATA/data_track_5/AIC21_Track5_NL_Retrieval"
VEH_DIR = '/content/SOURCE/nanonets_object_tracking/feature/vehicle'
COLOR_DIR = '/content/SOURCE/nanonets_object_tracking/feature/color'
SAVE_JSON_DIR = '/content/nanonets_object_tracking/json_result'
SAVE_VISUALIZE_DIR = '/content/nanonets_object_tracking/result'

def get_gt_from_idx(idx_image, gt_dict):
    frame_info = gt_dict[idx_image]
    key = list(frame_info.keys())[0]
    l = min(50, len(frame_info[key]))

    detections = []
    out_scores = []
    
    for i in range(l):
        x_0, y_0, x_1, y_1 = frame_info[key][i]
        x_0, y_0, x_1, y_1 = int(x_0), int(y_0), int(x_1), int(y_1)

        w = x_1 - x_0
        h = y_1 - y_0

        detections.append([x_0,y_0,w,h])
        out_scores.append(1)
    return detections,out_scores

def get_dict_track(filename):
    return json.load(open(filename))


def get_img_name(img_dict):
    ans = []
    l = len(img_dict)
    for i in range(l):
        name = list(img_dict[i].keys())[0]
        ans.append(name)
    return ans

def get_box_features_in_image(veh_features, col_features, img_name):
    features = []
    for (col_feat, veh_feat) in zip(col_features[img_name], veh_features[img_name]):
        feature = torch.cat((col_feat, veh_feat), 1)
        feature = feature.squeeze().detach().cpu().numpy()
        features.append(feature)
    return features

def json_dump(data: dict, save_path: str, verbose: bool=True):
    with open(col_fail_save_path, 'w') as f:
        json.dump(data, f, indent=2)
    if verbose:
        print(f'Save data to {save_path}')
    pass

def print_fail_dict(data, mode='VEHICLE'):
    print(f'{mode} fail features')
    for track_id in data.keys():
        print(f'{track_id}: {len(data[track_id])}')
    pass

def scan_data(track_keys, gt_dict):
    # Check extracted features (exist or not)
    fail_col, fail_veh = {}, {}

    for track_key in tqdm(track_keys):
        track_key = os.path.splitext(track_key)[0]
        
        fail_col[track_key], fail_veh[track_key] = [], []
        img_dict = gt_dict[track_key]
        img_names = get_img_name(img_dict)
        
        veh_path = os.path.join(VEH_DIR, f"{track_key}.pickle")
        with open(veh_path, 'rb') as handle:
            veh_features = pickle.load(handle)[track_key]
        
        color_path = os.path.join(COLOR_DIR, f'{track_key}.pickle')
        with open(color_path, 'rb') as handle:
            col_features = pickle.load(handle)[track_key]
        
        for img_name in img_names:
            img_col_feat, img_veh_feat = col_features.get(img_name), veh_features.get(img_name)
            if img_col_feat is None:
                fail_col[track_key].append(img_name)
            if img_veh_feat is None:
                fail_veh[track_key].append(img_name)

    col_fail_save_path = osp.join(SAVE_JSON_DIR, 'fail_col_feats.json')
    veh_fail_save_path = osp.join(SAVE_JSON_DIR, 'fail_veh_feats.json')
    json_dump(fail_col, col_fail_save_path)
    json_dump(fail_veh, veh_fail_save_path)
    print_fail_dict(fail_col)
    print_fail_dict(fail_veh)
    pass

if __name__ == '__main__':
	#Load detections for the video. Options available: yolo,ssd and mask-rcnn
	# filename = 'det/det_ssd512.txt'
    filename_track = TRACK_DIR

    gt_dict = get_dict_track(filename_track)

    text_file = open("/content/nanonets_object_tracking/keys.txt", "r")
    track_keys = text_file.read().split('\n')
    track_keys = track_keys[:len(track_keys)-1]

    if COLAB:
        print('>> Scanning box features')
        scan_data(track_keys, gt_dict)

    print('>> Run DeepSort')
    for track_key in tqdm(track_keys):
        track_key = os.path.splitext(track_key)[0]
        img_dict = gt_dict[track_key]
        img_names = get_img_name(img_dict)
        
        veh_path = os.path.join(VEH_DIR, f"{track_key}.pickle")
        with open(veh_path, 'rb') as handle:
            veh_features = pickle.load(handle)[track_key]
        
        color_path = os.path.join(COLOR_DIR, f'{track_key}.pickle')
        with open(color_path, 'rb') as handle:
            col_features = pickle.load(handle)[track_key]

        # cap = cv2.VideoCapture('vdo.avi')
        #an optional mask for the given video, to focus on the road. 
        # mask = get_mask('roi.jpg')

        ans = {}
        #Initialize deep sort.
        deepsort = deepsort_rbc()

        # frame_id = 1

        # mask = np.expand_dims(mask,2)
        # mask = np.repeat(mask,3,2)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_visualize_path = os.path.join(SAVE_VISUALIZE_DIR, f'{track_key}.avi')
        out = cv2.VideoWriter(save_visualize_path,fourcc, 2, (1920,1080))

        for i in range(len(img_names)):
            img_path = os.path.join(ROOT_DIR, img_names[i])
            frame = cv2.imread(img_path)
            frame = frame.astype(np.uint8)

            detections,out_scores = get_gt_from_idx(i, gt_dict[track_key])

        # while True:
        # 	print(frame_id)		

        # 	ret,frame = cap.read()
        # 	if ret is False:
        # 		frame_id+=1
        # 		break	

        # 	frame = frame * mask
        # 	frame = frame.astype(np.uint8)

        # 	detections,out_scores = get_gt(frame,frame_id,gt_dict)

        # 	if detections is None:
        # 		print("No dets")
        # 		frame_id+=1
        # 		continue

            detections = np.array(detections)
            out_scores = np.array(out_scores) 
            
            features = get_box_features_in_image(veh_features, col_features, img_names[i])
            
            tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections,features)

            track_list = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr() #Get the corrected/predicted bounding box
                id_num = str(track.track_id) #Get the ID for the particular track.
                features = track.features #Get the feature vector corresponding to the detection.

                track_dict = {}
                track_dict["id"] = id_num
                track_dict["box"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                track_dict["feature"] = features
                track_list.append(track_dict)
                #Draw bbox from tracker.
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

                #Draw bbox from detector. Just to compare.
                for det in detections_class:
                    bbox = det.to_tlbr()
                    cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
            
            ans[img_names[i]] = track_list
            # cv2.imshow('frame',frame)
            out.write(frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            # 	break
            # frame_id+=1
        save_json_path = os.path.join(SAVE_JSON_DIR, f'{track_key}.pickle')
        with open(save_json_path, 'wb') as handle:
            pickle.dump(ans, handle, protocol=pickle.HIGHEST_PROTOCOL)
        out.release()



'''
def get_dict(filename):
	with open(filename) as f:	
		d = f.readlines()

	d = list(map(lambda x:x.strip(),d))

	last_frame = int(d[-1].split(',')[0])

	gt_dict = {x:[] for x in range(last_frame+1)}

	for i in range(len(d)):
		a = list(d[i].split(','))
		a = list(map(float,a))	

		coords = a[2:6]
		confidence = a[6]
		gt_dict[a[0]].append({'coords':coords,'conf':confidence})

	return gt_dict

def get_mask(filename):
	mask = cv2.imread(filename,0)
	mask = mask / 255.0
	return mask
'''

'''
def get_gt(image,frame_id,gt_dict):
	if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
		return None,None,None

	frame_info = gt_dict[frame_id]

	detections = []
	ids = []
	out_scores = []
	for i in range(len(frame_info)):

		coords = frame_info[i]['coords']

		x1,y1,w,h = coords
		x2 = x1 + w
		y2 = y1 + h

		xmin = min(x1,x2)
		xmax = max(x1,x2)
		ymin = min(y1,y2)
		ymax = max(y1,y2)	

		detections.append([x1,y1,w,h])
		out_scores.append(frame_info[i]['conf'])

	return detections,out_scores
'''