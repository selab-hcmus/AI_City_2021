import os 
import os.path as osp
from .utils import SAVE_DIR


tracking_config = {
    'METRIC': {'NAME': 'cosine', 'THRESHOLD': 0.3, 'BUDGET': 70},
    'TRACKER': {'MAX_IOU_DISTANCE': 0.7, 'MAX_AGE': 10, 'N_INIT': 1},
    'ATTENTION_THRES': 0.3,

    # Attention mask
    'ATN_MASK': {'EXPAND_RATIO': 0.35, 'N_EXPAND': 2},

    # Save dir
    'MASK_SAVE_DIR': osp.join(SAVE_DIR, 'attn_mask'),
    'VIS_SAVE_DIR': osp.join(SAVE_DIR, 'vis_tracking_result'),
    'TRACK_SAVE_DIR': osp.join(SAVE_DIR, 'tracking_result'),
}

subject_config = {
    'IOU_ACCEPT_THRES': 0.2, 'SCORE_THRES': 0.75, 'IOU_AVG_THRES': 0.75,
    'COMPARE_RANGE': 30
}

stop_config = {
    'STOP_IOU_THRES': 0.4, 'COMPARE_RANGE': 3
}

class_config = {
    'VEH_THRES': 0.6, 'COL_THRES': 0.3
}