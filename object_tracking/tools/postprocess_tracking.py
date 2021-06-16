import os 
import os.path as osp
from tqdm import tqdm 
import sys 

from utils import create_logger
from utils import json_load, prepare_dir
from utils.data_manager import test_track_map, TEST_TRACK_JSON, RESULT_DIR, DATA_DIR
from object_tracking.library import VideoResult

from object_tracking.utils import (
    find_stop_track, xywh_to_xyxy, find_subject_track, SAVE_DIR,
    subject_config, stop_config, tracking_config, class_config
)
from classifier import ClassifierManager

VISUALIZE = True
def main():
    post_process_save_dir = osp.join(SAVE_DIR, 'final_json')
    vis_post_process_save_dir = osp.join(SAVE_DIR, 'vis_final_json')
    prepare_dir(post_process_save_dir)
    prepare_dir(vis_post_process_save_dir)

    track_save_dir = tracking_config['TRACK_SAVE_DIR']
    test_data = json_load(TEST_TRACK_JSON)

    # Init classifier for veh/col prediction
    class_manager = ClassifierManager(cuda=True, load_ckpt=True, eval=True)
    print(f'Init classifier successfully')

    # logger = create_logger(osp.join(SAVE_DIR, 'class_prediction.log'))
    # test_ids = [0]
    for fname in tqdm(os.listdir(track_save_dir)):
        fpath = osp.join(track_save_dir, fname)
        new_id = fname.split('.')[0]
        
        # if int(new_id) not in test_ids:
        #     continue
        # else:
        #     print(f'Test on id {new_id}')

        old_id = test_track_map[new_id]
        gt_boxes = test_data[old_id]['boxes']
        gt_boxes = [xywh_to_xyxy(box) for box in gt_boxes]
        vid_data = VideoResult(fpath)

        # 1. Find subject 
        subject_track_id, score_dict = find_subject_track(vid_data, gt_boxes)
        vid_data.set_subject(subject_track_id)

        # 2. Find stop vehicles
        stop_tracks = find_stop_track(vid_data)
        vid_data.set_stop_tracks(stop_tracks)

        save_path = osp.join(post_process_save_dir, f'{new_id}.json')
        vis_save_path = osp.join(vis_post_process_save_dir, f'{new_id}.avi')

        # 3. set class name
        vid_data.set_class_names(class_manager, veh_thres=class_config['VEH_THRES'], col_thres=class_config['COL_THRES'])
        vid_data.to_json(save_path)
        if VISUALIZE:
            vid_data.visualize(vis_save_path)
        pass
    pass 

if __name__ == '__main__':
    main()
    pass 
