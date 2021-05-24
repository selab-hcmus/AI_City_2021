import pickle, json, os 
import os.path as osp 
from tqdm import tqdm

RESULT_DIR = './results'
TRAIN_RESULTS = osp.join(RESULT_DIR, 'annotate_train')
TEST_RESULTS = osp.join(RESULT_DIR, 'annotate_test')
SAVE_DIR = './results'

data_dir = {
    'train_input': TRAIN_RESULTS,
    'test_input': TEST_RESULTS,
    'train_save': osp.join(SAVE_DIR, 'annotate_time_train'),
    'test_save': osp.join(SAVE_DIR, 'annotate_time_test')
}
# os.makedirs(data_dir['train_save'], exist_ok=True)
# os.makedirs(data_dir['test_save'], exist_ok=True)


def convert_video_track(video_data, save_path: str=None):
    res = {}
    list_frames = list(video_data.keys())
    n_frames = len(list_frames)
    frame_ids = list(range(n_frames))
    tracks_map = {}
    
    for frame, i in zip(list_frames, frame_ids):
        if len(video_data[frame]) == 0:
            continue
        for track_item in video_data[frame]:
            if tracks_map.get(track_item['id']) is None:
                tracks_map[track_item['id']] = {
                    'frame_order': [],
                    'boxes': [],
                    'vehicle_type': [],
                    'color': [],
                }
            
            tracks_map[track_item['id']]['frame_order'].append(i)
            tracks_map[track_item['id']]['boxes'].append(track_item['box'])
            # return
            pass
        pass

    res['list_frames'] = list_frames
    res['n_frames'] = n_frames 
    res['frame_ids'] = frame_ids,
    res['n_tracks'] = len(list(tracks_map.keys()))
    res['track_map'] = tracks_map

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(res, f, indent=2)

    return res 

def convert(mode):
    for filename in tqdm(os.listdir(data_dir[f'{mode}_input'])):
        track_path = osp.join(data_dir[f'{mode}_input'], filename)
        save_path = osp.join(data_dir[f'{mode}_save'], filename.replace('.pkl', '.json'))
        video_data = pickle.load(open(track_path, 'rb'))
        convert_video_track(video_data, save_path)
        # return
        pass
    pass

# if __name__ == '__main__':
#     convert(mode='train')
#     convert(mode='test')
#     pass