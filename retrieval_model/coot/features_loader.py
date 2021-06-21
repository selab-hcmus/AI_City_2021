"""
Feature loading.
"""
import json
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pickle
from tqdm import tqdm

from nntrainer.utils_torch import create_shared_array


class VideoFeatureLoader(object):
    """
    Helper class to load video features (h5) format.

    Args:
        dataset_path: Dataset path.
        features_name: Name to identify the features.
        features_source: Type of files (h5)
        data_keys: List of data keys to load. Important, these are video ids instead of datapoints ids.
        preload_vid_feat: Cache video features to memory.
    """
    def __init__(
            self, dataset_path: Path, features_name: str, features_source: str, data_keys: List[str], *,
            preload_vid_feat: bool = False):

        self.dataset_path = dataset_path
        self.features_name = features_name
        self.features_source = features_source
        self.data_keys = data_keys
        self.cached_data = {}
        self.preload_vid_feat = preload_vid_feat

        self.num_frames = {}
        for key, data in tqdm(self.get_features_as_items(load_all=True), desc="Analyzing features"):
            self.num_frames[key] = int(data.shape[0])
    
    def get_features_by_key(self, item: str) -> np.ndarray:
        if self.features_source == "h5":
            # load from h5
            h5 = h5py.File((self.dataset_path / f"{self.features_name}.h5"), "r")
            return np.array(h5[item])

        raise NotImplementedError(f"Feature source type {self.features_source} not understood.")

    def get_features_as_items(self, *, load_all: bool = False):
        if self.features_source == "h5":
            # load from h5
            h5 = h5py.File((self.dataset_path / f"{self.features_name}.h5"), "r")
            if load_all:
                for key, data in h5.items():
                    yield key, data
            else:
                for key in self.data_keys:
                    yield key, h5[key]
        else:
            raise NotImplementedError(f"Feature source type {self.features_source} not understood.")
    
    def __getitem__(self, key: str) -> np.ndarray:
        assert key in self.data_keys, f"Video features for datapoint {key} not found."
        if self.preload_vid_feat:
            # return buffered data
            return self.cached_data[key]
        # read h5 and return (slow)
        return self.get_features_by_key(key)


class TextFeaturesLoader(object):
    def __init__(self, dataset_path: Path, features_name: str, features_source: str, keys: List[str],
                 *, preload_text_feat: bool = False):
        """
        Helper class to load text features (h5) format.

        Args:
            dataset_path: Dataset path.
            features_name: Identifier for text features.
            features_source: Type of files (h5, npz, lmdb, ...)
            keys: List of keys to load.
            preload_text_feat: Cache video features to memory.
        """
        assert features_source == "h5", f"Text feature source {features_source} not implemented."
        self.features_file = dataset_path / f"{features_name}.h5"
        self.sentence_splits_file = dataset_path / f"{features_name}_sentence_splits.json"
        self.data_keys = keys
        self.cached_data = {}
        self.preload_text_feat = preload_text_feat

        # load text features sentence splits
        self.sentence_splits = json.load(self.sentence_splits_file.open("rt", encoding="utf8"))

        if self.preload_text_feat:
            # buffer data to memory
            h5 = h5py.File(self.features_file, "r")
            for key in tqdm(self.data_keys, desc="Preloading text"):
                old_key = f"v_{key[:11]}"  # backwards compatible loading
                if key in h5:
                    self.cached_data[key] = create_shared_array(h5[key])
                elif old_key in h5:
                    self.cached_data[key] = create_shared_array(h5[old_key])
                else:
                    raise KeyError(f"Key {key} not found in {self.features_file}. Keys in the file look like this: "
                                   f"{list(h5.keys())[:10]}, ...")

    def __getitem__(self, key: str) -> Tuple[np.ndarray, List[int]]:
        """
        Load text features given the data id.

        Args:
            key: Data id of the video.

        Returns:
            Tuple of:
                Text features with shape (num_tokens, feat_dim).
                List of lengths for each sentence, this needed to split the entire paragraph of features
                    back into sentences.
        """
        assert key in self.data_keys, f"Text features for datapoint {key} not found."
        if self.preload_text_feat:
            # return buffered data
            text_feats = self.cached_data[key]
        else:
            # read h5 and return (slow)
            with h5py.File(self.features_file, "r") as h5:
                old_key = f"v_{key[:11]}"  # backwards compatible loading
                if key in h5:
                    text_feats = np.array(h5[key])
                elif old_key in h5:
                    key = old_key
                    text_feats = np.array(h5[old_key])
                else:
                    raise KeyError(f"Key {key} not found in {self.features_file}. Keys in the file look like this: "
                                   f"{list(h5.keys())[:10]}, ...")
        if key in self.sentence_splits:
            sent_cap_len_list = self.sentence_splits[key]
        else:
            old_key = f"v_{key[:11]}"  # backwards compatible loading
            sent_cap_len_list = self.sentence_splits[old_key]
        return text_feats, sent_cap_len_list


class TextActionLoader(object):
    def __init__(self, label_path: str):
        self.label_map = pickle.load(open(label_path, 'rb'))
        pass 
    
    def __getitem__(self, key: str):
        return self.label_map[key]
        
    pass
