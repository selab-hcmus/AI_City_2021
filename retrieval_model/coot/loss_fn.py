"""
Loss functions.
"""

from typing import Callable, Dict

import torch as th
from torch import nn

from nntrainer import typext
from nntrainer.typext import INF


class LossesConst(typext.ConstantHolder):
    CONTRASTIVE = "contrastive"
    CROSSENTROPY = "crossentropy"


def cosine_sim(visual_emb: th.Tensor, text_emb: th.Tensor) -> th.Tensor:
    """
    Calculate cosine similarity.

    Args:
        visual_emb: Visual embedding with shape (num_datapoints, dim_embedding)
        text_emb: Text embedding with shape (num_datapoints, dim_embedding)

    Returns:
        Cosine similariies with shape (num_datapoints, num_datapoints)
    """
    return visual_emb.mm(text_emb.t())


class ContrastiveLossConfig(typext.ConfigClass):
    """
    Contrastive loss Configuration Class

    Args:
        config: Configuration dictionary to be loaded, saving part.
    """

    def __init__(self, config: Dict) -> None:
        self.margin: float = config.pop("margin")
        self.weight_high: float = config.pop("weight_high")
        self.weight_high_internal: float = config.pop("weight_high_internal")
        self.weight_low: float = config.pop("weight_low")
        self.weight_low_internal: float = config.pop("weight_low_internal")
        self.weight_context: float = config.pop("weight_context")
        self.weight_context_internal: float = config.pop("weight_context_internal")


class ContrastiveLoss(nn.Module):
    """
    Regular Contrastive Loss between 2 groups of embeddings
    """
    def __init__(self, margin: float, max_violation: bool = False, norm: bool = True, use_cuda: bool = True):
        super().__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.norm = norm
        self.max_violation = max_violation
        self.use_cuda = use_cuda

    def forward(self, im, s):
        """
        Inputs shape (batch, embed_dim)

        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)

        Returns:
        """
        # compute image-sentence score matrix - how close is im(y) to s(x)
        scores = self.sim(im, s) # [batch, batch]
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals, where there is just the margin left
        mask: th.Tensor = th.eye(scores.shape[0]).bool()
        if self.use_cuda:
            mask = mask.cuda(non_blocking=True)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if self.norm:
            if self.max_violation:
                return (cost_s.sum() + cost_im.sum()).div(im.shape[0])
            else:
                return (cost_s.sum() + cost_im.sum()).div(im.shape[0] * s.shape[0])
        
        return (cost_s.sum() + cost_im.sum())#/(im.shape[0])


class BceDiceLoss(nn.Module):
    def __init__(self, weight_bce: float=0.0, weight_dice: float=1.0):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        pass 
    
    def forward(self, inp, target):
        batch_size = inp.shape[0]
        inp = inp.view(batch_size, -1)
        target = target.view(batch_size, -1)
        # bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inp, target).mean(dim=-1).double() #[B]
        dice_coef = (2.0*(inp*target).sum(dim=-1).double() + 1)/(
            inp.sum(dim=-1).double() + target.sum(dim=-1).double() + 1
        )
        dice_loss = 1-dice_coef
        # total_loss = th.mean(self.weight_bce*bce_loss + self.weight_dice*dice_loss)
        total_loss = th.mean(self.weight_dice*dice_loss)

        return total_loss
    pass


def iou_dice_score(pred, target):
    batch_size = pred.shape[0]
    inp = pred.view(batch_size, -1)
    target = target.view(batch_size, -1)
    inter = (inp*target).sum(dim=-1)
    union = (inp+target).sum(dim=-1) - inter 
    iou = (inter + 1)/(union + 1)
    dice = (2*inter + 1)/(union + inter + 1)

    return th.mean(iou), th.mean(dice)

def compute_mean_distance_l2(c, s):
    return th.mean((c - s) ** 2, dim=-1)

def compute_mean_distance_negative_l2(c, s):
    return -compute_mean_distance_l2(c, s)
