import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch as th
from torch.utils import data as th_data

import coot.configs_retrieval
from coot.configs_retrieval import RetrievalDatasetConfig
from coot.features_loader import TextFeaturesLoader, VideoFeatureLoader
from nntrainer import data as nn_data, data_text, maths, typext, utils, utils_torch

from coot.dataset_utils import (
    RetrievalDataPointTuple, RetrievalDataBatchTuple,
    TextDataBatchTuple, VideoDataBatchTuple,
    TextDataPointTuple, VideoDataPointTuple,
    SPLIT_QUERY_IDS, TRAIN_TRACK_JSON, TEST_TRACK_JSON, TEST_QUERY_JSON
)

class VideoDataset(th_data.Dataset):
    """
    Dataset for testing retrieval.

    Args:
        cfg: Dataset configuration class.
        path_data: Dataset base path.
        verbose: Print output (cannot use logger in multiprocessed torch Dataset class)
    """

    def __init__(self, cfg: RetrievalDatasetConfig, path_data: Union[str, Path], *, verbose: bool = False):
        # store config
        self.path_data = Path(path_data)
        self.cfg = cfg
        self.verbose = verbose
        self.is_train = False
        
        test_dict = json.load(open(TEST_TRACK_JSON))
        self.data_keys = list(test_dict.keys())

        # setup paths
        self.path_dataset = self.path_data 

        # For each key (datapoint) get the data_key (reference to the video file).
        # A single video can appear in multiple datapoints.
        # self.data_keys = self.keys # [raw_meta[key]["data_key"] for key in self.keys]

        # load video features
        self.vid_feats = VideoFeatureLoader(
            self.path_dataset, self.cfg.vid_feat_name, self.cfg.vid_feat_source, self.data_keys,
            preload_vid_feat=self.cfg.preload_vid_feat)

        # load preprocessing function for text
        self.text_preproc_func = data_text.get_text_preprocessor(self.cfg.text_preprocessing)

    def get_vid_frames_by_indices(self, key: str, indices: List[int]) -> np.ndarray:
        data_key = self.meta[key]["data_key"]
        return self.vid_feats[data_key][indices]

    def get_vid_feat_by_amount(self, key: str, num_frames: int) -> np.ndarray:
        indices = maths.compute_indices(self.vid_feats.num_frames[key], num_frames, self.is_train)
        return self.vid_feats[key][indices]

    def get_clip_frames_by_amount(self, key: str, seg_num: int, num_frames: int) -> np.ndarray:
        seg = self.meta[key]["segments"][seg_num]
        indices = maths.compute_indices(seg["num_frames"], num_frames, self.is_train)
        indices += seg["start_frame"]
        return self.get_vid_frames_by_indices(key, indices)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, item: int) -> VideoDataPointTuple:
        key = self.keys[item]
        data_key = key
        vid_feat = self.vid_feats[data_key]

        # ---------- load video frames features ----------
        vid_feat_len = vid_feat.shape[0]
        if vid_feat_len > self.cfg.max_frames:
            vid_feat_len = self.cfg.max_frames 
        vid_feat = th.Tensor(self.get_vid_feat_by_amount(key, vid_feat_len))
        
        assert vid_feat_len == int(vid_feat.shape[0]), \
            f'vid_feat_len != vid_feat.shape[0]: {vid_feat_len} != {int(vid_feat.shape[0])}'

        if self.cfg.frames_noise != 0:
            # add noise to frames if needed
            vid_frames_noise = utils_torch.get_truncnorm_tensor(vid_feat.shape, std=self.cfg.frames_noise)
            vid_feat += vid_frames_noise

        # return single datapoint
        return VideoDataPointTuple(
            key, data_key,
            vid_feat, vid_feat_len, 
        )

    def collate_fn(self, data_batch: List[VideoDataPointTuple]):
        """
        Collate the single datapoints above. Custom collation needed since sequences have different length.

        Returns:
        """
        batch_size = len(data_batch)
        key: List[str] = [d.key for d in data_batch]
        data_key: List[str] = [d.data_key for d in data_batch]

        # ---------- collate video features ----------

        # read video features list
        list_vid_feat = [d.vid_feat for d in data_batch] #[(num_feat, vid_feat_dim), ...]
        vid_feat_dim: int = list_vid_feat[0].shape[-1]

        # read video sequence lengths
        list_vid_feat_len = [d.vid_feat_len for d in data_batch]
        vid_feat_len = th.Tensor(list_vid_feat_len).long()
        vid_feat_max_len = int(vid_feat_len.max().numpy())

        # put all video features into a batch, masking / padding as necessary
        vid_feat = th.zeros(batch_size, vid_feat_max_len, vid_feat_dim).float()
        vid_feat_mask = th.ones(batch_size, vid_feat_max_len).bool()
        for batch, (seq_len, item) in enumerate(zip(list_vid_feat_len, list_vid_feat)):
            vid_feat[batch, :seq_len] = item
            vid_feat_mask[batch, :seq_len] = 0
        
        ret = VideoDataBatchTuple(
            key, data_key,
            vid_feat, vid_feat_mask, vid_feat_len, 
        )
        return ret


class TextDataset(th_data.Dataset):
    """
    Dataset for testing retrieval.

    Args:
        cfg: Dataset configuration class.
        path_data: Dataset base path.
        verbose: Print output (cannot use logger in multiprocessed torch Dataset class)
    """
    def __init__(self, cfg: RetrievalDatasetConfig, path_data: Union[str, Path], *, verbose: bool = False):
        # store config
        self.path_data = Path(path_data)
        self.cfg = cfg
        self.verbose = verbose
        self.is_train = False

        self.test_dict = json.load(open(TEST_QUERY_JSON))
        self.data_keys = list(self.test_dict.keys())
        
        # setup paths
        self.path_dataset = self.path_data #/ self.cfg.name
        # For each key (datapoint) get the data_key (reference to the video file).
        # A single video can appear in multiple datapoints.
        # self.data_keys = self.keys # [raw_meta[key]["data_key"] for key in self.keys]

        # load text features
        self.text_feats = TextFeaturesLoader(
            self.path_dataset, f"{self.cfg.text_feat_name}", self.cfg.text_feat_source, self.keys,
            preload_text_feat=self.cfg.preload_text_feat)
        # load preprocessing function for text
        self.text_preproc_func = data_text.get_text_preprocessor(self.cfg.text_preprocessing)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, item: int) -> RetrievalDataPointTuple:
        key = self.keys[item]
        data_key = key

        # ---------- load text as string ----------
        sentences = self.text_preproc_func(self.test_dict[key])

        # ---------- load text features ----------
        par_feat = self.text_feats[data_key][0]
        par_feat_len = int(par_feat.shape[0])
        par_feat = th.Tensor(par_feat).float()

        # return single datapoint
        return TextDataPointTuple(
            key, data_key, sentences, 
            par_feat, par_feat_len, 
        )

    def collate_fn(self, data_batch: List[TextDataPointTuple]):
        batch_size = len(data_batch)
        key: List[str] = [d.key for d in data_batch]
        data_key: List[str] = [d.data_key for d in data_batch]

        # store input text: for each video, each sentence, store each word as a string
        sentences: List[str] = [d.sentences for d in data_batch]

        # ---------- collate paragraph features ----------

        # read paragraph features list
        list_par_feat = [d.par_feat for d in data_batch]
        par_feat_dim: int = list_par_feat[0].shape[-1]

        # read paragraph sequence lengths
        list_par_feat_len = [d.par_feat_len for d in data_batch]
        par_feat_len = th.Tensor(list_par_feat_len).long()
        par_feat_max_len = int(par_feat_len.max().numpy())

        # put all paragraph features into a batch, masking / padding as necessary
        par_feat = th.zeros(batch_size, par_feat_max_len, par_feat_dim).float()
        par_feat_mask = th.ones(batch_size, par_feat_max_len).bool()
        for batch, (seq_len, item) in enumerate(zip(list_par_feat_len, list_par_feat)):
            par_feat[batch, :seq_len, :] = item
            par_feat_mask[batch, :seq_len] = 0

        
        ret = TextDataBatchTuple(
            key, data_key, sentences, 
            par_feat, par_feat_mask, par_feat_len
        )
        return ret

