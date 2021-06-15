from tqdm import tqdm
import os 
import os.path as osp

from object_tracking.library import VideoResult
from detector.library import StopDetector
from utils import RESULT_DIR, dict_save

track_dir = osp.join(RESULT_DIR, 'object_tracking_exp/test_deepsort_v4-3/json_full') 
save_dir = osp.join(RESULT_DIR, 'detector')
os.makedirs(save_dir, exist_ok=True)

config = {
    'k': 5, 'delta': 5, 'alpha': 0.15
}

def main1():
    """Run on original tracks
    """
    detector = StopDetector(**config)
    stop_ids = []
    fail_ids = []
    for fname in tqdm(os.listdir(track_dir)):
        fpath = osp.join(track_dir, fname)
        track_id = fname.split('.')[0]
        vid_data = VideoResult(fpath)
        if vid_data.subject is None:
            fail_ids.append(track_id)
            continue
        subject_data = vid_data.get_subject()
        subject_boxes = subject_data.boxes

        is_stop = detector.process(subject_boxes)
        if is_stop:
            stop_ids.append(track_id)
        pass
    
    res = {'config': config, 'stop_ids': stop_ids, 'fail_ids': fail_ids}
    save_path = osp.join(save_dir, 'stop.json')
    dict_save(res, save_path)
    pass

def main2():
    """Run on tracking result
    """
    pass 

if __name__ == '__main__':
    main1()