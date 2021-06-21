import os 
import os.path as osp

from utils import RESULT_DIR, DATA_DIR, prepare_dir


SAVE_DIR = prepare_dir(osp.join(RESULT_DIR, 'retrieval_model'))
