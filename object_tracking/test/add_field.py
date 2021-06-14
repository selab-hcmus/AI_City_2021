import os 
import os.path as osp
from tqdm import tqdm

from object_tracking.utils import EXP_SAVE_DIR
from object_tracking.library import VideoResult
from utils.data_manager import DATA_DIR
from utils import dict_load, dict_save

exp_id = 'test_deepsort_v4-3'
mask_exp_id = 'test_deepsort_v4-1'
stop_dir = osp.join(EXP_SAVE_DIR, exp_id, 'json_stop')
save_dir = osp.join(EXP_SAVE_DIR, exp_id, 'json_full')
mask_dir = osp.join(EXP_SAVE_DIR, mask_exp_id, 'attn_mask/npy_1')

os.makedirs(save_dir, exist_ok=True)

for fname in tqdm(os.listdir(stop_dir)):
    track_id = fname.split('.')[0]
    fpath = osp.join(stop_dir, fname)
    mask_path = osp.join(mask_dir, f'{track_id}.npy')
    vid_data = VideoResult(fpath)
    vid_data.set_attn_mask(mask_path)
    
    save_path = osp.join(save_dir, f'{track_id}.json')
    vid_data.to_json(save_path, is_feat=False)
    pass
