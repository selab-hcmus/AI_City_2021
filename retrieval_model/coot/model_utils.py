
import torch as th
from torch.cuda.amp import autocast

from coot.configs_retrieval import RetrievalConfig, RetrievalNetworksConst
from nntrainer import models, typext


class RetrievalVisualEmbTuple(typext.TypedNamedTuple):
    """
    Definition of computed visual embeddings

    Notes:
        vid_emb: Video embedding with shape (batch, global_emb_dim)
        clip_emb: Clip embedding with shape (total_num_clips, local_emb_dim)
        vid_context: Video context with shape (batch, local_emb_dim)
        clip_emb_reshaped: Clip embeddings reshaped for input to the global model
            with shape (batch, max_num_clips, local_emb_dim)
        clip_emb_mask: Mask for the reshaped Clip embeddings with shape (batch, max_num_clips)
        clip_emb_lens: Lengths of the reshaped Clip embeddings with shape (batch)
    """
    # vid_emb: th.Tensor
    # clip_emb: th.Tensor
    vid_context: th.Tensor
    action_pred: th.Tensor
    # clip_emb_reshape: th.Tensor
    # clip_emb_mask: th.Tensor
    # clip_emb_lens: th.Tensor


class RetrievalTextEmbTuple(typext.TypedNamedTuple):
    """
    Definition of computed text embeddings:

    Notes:
        par_emb: Paragraph embedding with shape (batch, global_emb_dim)
        sent_emb: Sentence embedding with shape (total_num_sents, local_emb_dim)
        par_context: Paragraph context with shape (batch, local_emb_dim)
        sent_emb_reshaped: Sentence embeddings reshaped for input to the global model
            with shape (batch, max_num_sents, local_emb_dim)
        sent_emb_mask: Mask for the reshaped sentence embeddings with shape (batch, max_num_sents)
        sent_emb_lens: Lengths of the reshaped sentence embeddings with shape (batch)
    """
    # par_emb: th.Tensor
    # sent_emb: th.Tensor
    par_context: th.Tensor
    # sent_emb_reshape: th.Tensor
    # sent_emb_mask: th.Tensor
    # sent_emb_lens: th.Tensor
