"""
Model manager for retrieval.

COOT is 4 times dlbase.models.TransformerLegacy
"""

import torch as th
from torch.cuda.amp import autocast

from coot.aic_dataset import RetrievalDataBatchTuple as Batch
from coot.configs_retrieval import RetrievalConfig, RetrievalNetworksConst
from coot.model_utils import RetrievalTextEmbTuple, RetrievalVisualEmbTuple
from nntrainer import models, typext


class RetrievalModelManager(models.BaseModelManager):
    """
    Interface to create the 2 coot models (vid + text) and do the forward pass.
    """

    def __init__(self, cfg: RetrievalConfig):
        super().__init__(cfg)
        # update config type hints
        self.cfg: RetrievalConfig = self.cfg


        self.model_dict[RetrievalNetworksConst.NET_VIDEO_LOCAL] = models.VideoEncoder(
            cfg.model_cfgs[RetrievalNetworksConst.NET_VIDEO_LOCAL],
            cfg.dataset_val.vid_feat_dim
        )
        self.model_dict[RetrievalNetworksConst.NET_TEXT_LOCAL] = models.TransformerLegacy(
            cfg.model_cfgs[RetrievalNetworksConst.NET_TEXT_LOCAL],
            cfg.dataset_val.text_feat_dim
        )

    def encode_visual(self, batch: Batch) -> RetrievalVisualEmbTuple:
        with autocast(enabled=self.is_autocast_enabled()):
            # reference models for easier usage
            net_vid_local = self.model_dict[RetrievalNetworksConst.NET_VIDEO_LOCAL]
            
            # compute video context
            vid_context, _, action_pred = net_vid_local(batch.vid_feat, batch.vid_feat_mask, batch.vid_feat_len, None)

            return RetrievalVisualEmbTuple(vid_context, action_pred)

    def encode_text(self, batch: Batch) -> RetrievalTextEmbTuple:
        with autocast(enabled=self.is_autocast_enabled()):
            # reference models for easier usage
            net_text_local = self.model_dict[RetrievalNetworksConst.NET_TEXT_LOCAL]
            
            # compute paragraph context
            par_context, _ = net_text_local(batch.par_feat, batch.par_feat_mask, batch.par_feat_len, None)
            
            return RetrievalTextEmbTuple(par_context)

