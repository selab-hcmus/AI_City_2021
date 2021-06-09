import os
import os.path as osp
from numpy.lib.npyio import save
import cv2
from tqdm import tqdm
import numpy as np

from utils import json_load, json_save, pickle_load, pickle_save
from object_tracking.utils import (
    ROOT_DIR,
    TRAIN_TRACK_DIR, TEST_TRACK_DIR, VEHCOL_FEAT_DIR, REID_FEAT_DIR,
    get_img_name, get_gt_from_idx, get_closest_box, 
)
from object_tracking.tools import convert_video_track, visualize
from object_tracking.test.test_utils import (
    SAVE_DIR
)
from object_tracking.deepsort import deepsort_rbc

save_dir = osp.join(SAVE_DIR, 'deepsort_v3')
save_json_dir = osp.join(save_dir, 'json')
save_vis_dir = osp.join(save_dir, 'video')
os.makedirs(save_json_dir, exist_ok=True)
os.makedirs(save_vis_dir, exist_ok=True)

ID_TO_COMPARE = [5, 6, 20, 34, 40, 64, 182, 188, 239, 310, 339, 349, 410, 436, 476]

def concat_feat(vehcol_feats: list, reid_feats: list):
    new_feats = []
    for feat_a, feat_b in zip(vehcol_feats, reid_feats):
        new_feats.append(np.concatenate([feat_a, feat_b], axis=0))

    return new_feats


def tracking(config: dict, json_save_dir: str, vis_save_dir: str, verbose=True):
    mode_json_dir = json_save_dir
    gt_dict = json_load(config["track_dir"])
    print(f'>> Run DeepSort on {config["mode"]} mode, save result to {mode_json_dir}')

    for track_order in ID_TO_COMPARE:
        track_order = str(track_order)
        if verbose:
            print(f'tracking order {track_order}')
        img_dict = gt_dict[track_order]
        img_names = get_img_name(img_dict)
        
        vehcol_feat_path = osp.join(VEHCOL_FEAT_DIR, f'{track_order}.pkl')
        reid_feat_path = osp.join(REID_FEAT_DIR, f'{track_order}.pkl')
        vehcol_feat = pickle_load(vehcol_feat_path)
        reid_feat = pickle_load(reid_feat_path)

        ans = {}
        #Initialize deep sort.
        deepsort = deepsort_rbc()

        if config["save_video"]:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            save_visualize_path = os.path.join(vis_save_dir, f'{track_order}.avi')
            out = None

        for i in tqdm(range(len(img_names))):
            img_path = osp.join(ROOT_DIR, img_names[i])

            frame = cv2.imread(img_path)
            frame = frame.astype(np.uint8)
            
            if i == 0 and config["save_video"]:
                h, w, c = frame.shape
                out = cv2.VideoWriter(save_visualize_path,fourcc, 1, (w,h))

            detections, out_scores = get_gt_from_idx(i, gt_dict[track_order])
            detections = np.array(detections)
            out_scores = np.array(out_scores)
            
            vehcol_features = vehcol_feat[img_names[i]]
            reid_features = reid_feat[img_names[i]]
            new_feats = concat_feat(vehcol_features, reid_features)

            features = new_feats
            
            tracker, detections_class = deepsort.run_deep_sort(out_scores, detections, features)

            track_list = []
            count = 0

            for track in tracker.tracks:
                count += 1
                if not track.is_confirmed() or track.time_since_update > 2:
                    continue

                bbox = track.to_tlbr() #Get the corrected/predicted bounding box
                id_num = str(track.track_id) #Get the ID for the particular track.
                features = track.features #Get the feature vector corresponding to the detection.

                track_dict = {"id": id_num}

                ans_box = get_closest_box(detections_class, bbox)
                bbox = ans_box
                # track_dict["box"] = [int(ans_box[0]), int(ans_box[1]), int(ans_box[2]), int(ans_box[3])]
                track_dict["box"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                track_dict["feature"] = features
                track_list.append(track_dict)

                if config["save_video"]:
                    #Draw bbox from tracker.
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

                    #Draw bbox from detector. Just to compare.
                    for det in detections_class:
                        bbox = det.to_tlbr()
                        cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
            
            ans[img_names[i]] = track_list

            if config["save_video"]:
                out.write(frame)
        
        if config["save_video"]:
            out.release()
        
        save_json_path = os.path.join(mode_json_dir, f'{track_order}.json')
        reformat_res = convert_video_track(ans, save_json_path)

    pass

if __name__ == '__main__':
    config = {
        "train_total": {
            "track_dir": TRAIN_TRACK_DIR,
            "mode": "train_total",
            "save_video": True,
        },
    }
    tracking(config['train_total'], save_json_dir, save_vis_dir, verbose=True)
