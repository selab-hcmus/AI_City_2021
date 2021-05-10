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
    Interface to create the 4 coot models and do the forward pass.
    """

    def __init__(self, cfg: RetrievalConfig):
        super().__init__(cfg)
        # update config type hints
        self.cfg: RetrievalConfig = self.cfg

        # find out input dimensions to the network
        input_dims = {
            RetrievalNetworksConst.NET_VIDEO_LOCAL: cfg.dataset_val.vid_feat_dim,
            RetrievalNetworksConst.NET_VIDEO_GLOBAL: cfg.model_cfgs[RetrievalNetworksConst.NET_VIDEO_LOCAL].output_dim,
            RetrievalNetworksConst.NET_TEXT_LOCAL: cfg.dataset_val.text_feat_dim,
            RetrievalNetworksConst.NET_TEXT_GLOBAL: cfg.model_cfgs[RetrievalNetworksConst.NET_TEXT_LOCAL].output_dim,
        }

        # create the 4 networks
        for key in RetrievalNetworksConst.values():
            # load model config
            current_cfg: models.TransformerConfig = cfg.model_cfgs[key]
            # create the network
            if current_cfg.name == models.TransformerTypesConst.TRANSFORMER_LEGACY:
                # old transformer
                self.model_dict[key] = models.TransformerLegacy(current_cfg, input_dims[key])
            else:
                raise NotImplementedError(f"Coot model type {current_cfg.name} undefined")

    def encode_visual(self, batch: Batch) -> RetrievalVisualEmbTuple:
        """
        Encode visual features to visual embeddings.

        Args:
            batch: Data batch.

        Returns:
            Video embeddings tuple.
        """
        with autocast(enabled=self.is_autocast_enabled()):
            # reference models for easier usage
            net_vid_local = self.model_dict[RetrievalNetworksConst.NET_VIDEO_LOCAL]
            
            # compute video context
            vid_context, _ = net_vid_local(batch.vid_feat, batch.vid_feat_mask, batch.vid_feat_len, None)

            # return RetrievalVisualEmbTuple(vid_emb, clip_emb, vid_context, clip_emb_reshape, clip_emb_mask, clip_emb_lens)
            return RetrievalVisualEmbTuple(vid_context)


    def encode_text(self, batch: Batch) -> RetrievalTextEmbTuple:
        """
        Encode text features to text embeddings.

        Args:
            batch: Batch data.

        Returns:
            Text embeddings tuple.
        """
        with autocast(enabled=self.is_autocast_enabled()):
            # reference models for easier usage
            net_text_local = self.model_dict[RetrievalNetworksConst.NET_TEXT_LOCAL]
            
            # compute paragraph context
            par_context, _ = net_text_local(batch.par_feat, batch.par_feat_mask, batch.par_feat_len, None)
            
            return RetrievalTextEmbTuple(par_context)

