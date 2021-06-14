import os 
import os.path as osp
from tqdm import tqdm

from utils.data_manager import DATA_DIR
from object_tracking.tools import visualize, visualize_subject
from utils import json_load

json_dir = '/home/ntphat/projects/AI_City_2021/object_tracking/results_exp/test_deepsort_v4-1/json_subject'
save_dir = '/home/ntphat/projects/AI_City_2021/object_tracking/results_exp/test_deepsort_v4-1/subject_v1/video_all'
os.makedirs(save_dir, exist_ok=True)

for track_json in tqdm(os.listdir(json_dir)):
    track_id = track_json.split('.')[0]
    track_data = json_load(osp.join(json_dir, track_json))
    save_path = osp.join(save_dir, f'{track_id}.avi')
    if osp.isfile(save_path):
        continue
    if track_data['subject'] is not None:
        subject_boxes = track_data['track_map'][track_data['subject']]['boxes']

        visualize_subject(track_data, None, DATA_DIR, save_path, subject_boxes)
    pass

