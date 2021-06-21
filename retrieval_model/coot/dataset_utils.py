from typing import List, Tuple, Union
import torch as th
import json 

from utils.data_manager import TRAIN_TRACK_JSON, TEST_TRACK_JSON, TEST_QUERY_JSON
from nntrainer import data as nn_data, data_text, maths, typext, utils, utils_torch


SPLIT_QUERY_IDS_JSON = './data/aic21/train_test_split_v1.json'
SPLIT_QUERY_IDS = json.load(open(SPLIT_QUERY_IDS_JSON, 'r'))

TEST_QUERY_JSON = '../dataset/data/test-queries.json'
ACTION_CLASS = 3
"""
Assumption:
- Video contains many clips

num_feat: total length of video
num_tokens: number of words in the given pargraph
num_feat_clip: length of clip
num_tokens_clip: number of words in the given clip
vid_feat_len: = num_feat
total_num_clips: 
"""

class RetrievalDataPointTuple(typext.TypedNamedTuple):
    """
    Definition of a single datapoint.
    """
    key: str
    data_key: str
    sentences: List[str]
    
    vid_feat: th.Tensor  # shape (num_feat, vid_feat_dim)
    vid_feat_len: int
    
    par_feat: th.Tensor  # shape (num_tokens, text_feat_dim)
    par_feat_len: int

    text_act: th.Tensor

    # clip_num: int
    # clip_feat_list: List[th.Tensor]  # shapes (num_feat_clip, vid_feat_dim)
    # clip_feat_len_list: List[int]
    # sent_num: int
    # sent_feat_list: List[th.Tensor]  # shapes (num_tokens_sent, text_feat_dim)
    # sent_feat_len_list: List[int]

    # shape tests for tensors
    _shapes_dict = {
        "vid_feat": (None, None),
        "par_feat": (None, None),
        # "clip_feat_list": (None, None),
        # "sent_feat_list": (None, None)
    }


class RetrievalDataBatchTuple(typext.TypedNamedTuple):
    """
    Definition of a batch.
    """
    key: List[str]
    data_key: List[str]
    sentences: List[List[str]]
    
    vid_feat: th.Tensor  # shape (batch_size, max_num_feat, vid_feat_dim) dtype float
    vid_feat_mask: th.Tensor  # shape (batch_size, max_num_feat) dtype bool
    vid_feat_len: th.Tensor  # shape (batch_size) dtype long
    
    par_feat: th.Tensor  # shape (batch_size, max_num_tokens, text_feat_dim) dtype float
    par_feat_mask: th.Tensor  # shape (batch_size, max_num_tokens) dtype bool
    par_feat_len: th.Tensor  # shape (batch_size) dtype long
    
    text_act: th.Tensor # shape (batch_size, 3)
    # clip_num: th.Tensor  # shape (batch_size) dtype long
    # clip_feat: th.Tensor  # shapes (total_num_clips, max_num_feat_clip, vid_feat_dim) dtype float
    # clip_feat_mask: th.Tensor  # shapes (total_num_clips, max_num_feat_clip) dtype bool
    # clip_feat_len: th.Tensor  # shapes (total_num_clips) dtype long
    # sent_num: th.Tensor  # shape (batch_size) dtype long
    # sent_feat: th.Tensor  # shapes (total_num_sents, max_num_feat_sent, text_feat_dim) dtype float
    # sent_feat_mask: th.Tensor  # shapes (total_num_sents, max_num_feat_sent) dtype bool
    # sent_feat_len: th.Tensor  # shapes (total_num_sents) dtype long

    # shape tests for tensors
    _shapes_dict = {
        "vid_feat": (None, None, None),
        "vid_feat_mask": (None, None),
        "vid_feat_len": (None,),
        "par_feat": (None, None, None),
        "par_feat_mask": (None, None),
        "par_feat_len": (None,),

        # "clip_num": (None,),
        # "clip_feat": (None, None, None),
        # "clip_feat_mask": (None, None),
        # "clip_feat_len": (None,),
        # "sent_num": (None,),
        # "sent_feat": (None, None, None),
        # "sent_feat_mask": (None, None),
        # "sent_feat_len": (None,),
    }


class TextDataPointTuple(typext.TypedNamedTuple):
    """
    Definition of a single datapoint.
    """
    key: str
    data_key: str
    sentences: List[str]

    par_feat: th.Tensor  # shape (num_tokens, text_feat_dim)
    par_feat_len: int

    # shape tests for tensors
    _shapes_dict = {
        "par_feat": (None, None),
    }

class VideoDataPointTuple(typext.TypedNamedTuple):
    """
    Definition of a single datapoint.
    """
    key: str
    data_key: str
    
    vid_feat: th.Tensor  # shape (num_feat, vid_feat_dim)
    vid_feat_len: int
    
    # shape tests for tensors
    _shapes_dict = {
        "vid_feat": (None, None),
    }


class TextDataBatchTuple(typext.TypedNamedTuple):
    """
    Definition of a batch.
    """
    key: List[str]
    data_key: List[str]
    sentences: List[List[str]]
    
    par_feat: th.Tensor  # shape (batch_size, max_num_tokens, text_feat_dim) dtype float
    par_feat_mask: th.Tensor  # shape (batch_size, max_num_tokens) dtype bool
    par_feat_len: th.Tensor  # shape (batch_size) dtype long
    
    # shape tests for tensors
    _shapes_dict = {
        "par_feat": (None, None, None),
        "par_feat_mask": (None, None),
        "par_feat_len": (None,),
    }


class VideoDataBatchTuple(typext.TypedNamedTuple):
    """
    Definition of a batch.
    """
    key: List[str]
    data_key: List[str]
    
    vid_feat: th.Tensor  # shape (batch_size, max_num_feat, vid_feat_dim) dtype float
    vid_feat_mask: th.Tensor  # shape (batch_size, max_num_feat) dtype bool
    vid_feat_len: th.Tensor  # shape (batch_size) dtype long
    
    # shape tests for tensors
    _shapes_dict = {
        "vid_feat": (None, None, None),
        "vid_feat_mask": (None, None),
        "vid_feat_len": (None,),
    }