"""
Transformer implementation.

Similar inference speed to pytorch built-in transformers.
"""

from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

import nntrainer.trainer_configs
import nntrainer.typext
import nntrainer.utils_torch
from nntrainer.initialization import init_network
from nntrainer.models.activations import ActivationConfig, ActivationConst, make_activation_module
from nntrainer.models.encoder import make_encoder_module
from nntrainer.models.mlp import MLP
from nntrainer.models.normalizations import NormalizationConfig, NormalizationConst, make_normalization_module
from nntrainer.models.poolers import PoolerConst, make_pooler_module
from nntrainer.models.transformer_cfg import TransformerConfig, TransformerEncoderConfig

# ---------- Module Implementations ----------
ACTION_HEAD = 3

class TransformerLegacy(nn.Module):
    """
    The COOT transformer (In total there are 4 of these.)
    """

    def __init__(self, cfg: TransformerConfig, feature_dim: int):
        super().__init__()
        error_txt = f"Transformer construction error: feature_dim "\
                    f"{feature_dim}. set output_dim of network before."
        assert feature_dim is not None, error_txt
        assert feature_dim > 0, error_txt
        self.input_dim = feature_dim
        self.cfg: TransformerConfig = cfg

        # dropout the input
        self.input_dropout = None
        if cfg.dropout_input > 0:
            self.input_dropout = nn.Dropout(cfg.dropout_input)

        # normalize input
        self.norm_input = make_normalization_module(self.input_dim, cfg.norm_input)

        # convert input with FC
        self.input_fc = None
        if cfg.use_input_fc:
            self.input_fc = MLP(feature_dim, cfg.input_fc_config)
            input_dim = cfg.input_fc_config.output_dim
        else:
            input_dim = feature_dim
        self.input_dim_transformed = input_dim

        # create local cls token adder (invidual CLS vector for each network)
        self.net_cls = None
        if cfg.add_local_cls_token:
            self.net_cls = LearnableClsToken(input_dim)

        # embed time information
        self.embedding = make_encoder_module(input_dim, cfg.positional_encoding)

        # self-attention transformer
        assert input_dim == cfg.selfatn.hidden_dim, (f"Input dim at this point of {input_dim} must match transformer"
                                                     f"dim of {cfg.selfatn.hidden_dim}")
        transformer_class = TransformerEncoder

        self.tf = transformer_class(self.cfg.selfatn)

        # build transformer for context
        self.use_context = cfg.use_context
        self.tf_context = None
        if self.use_context:
            self.tf_context = transformer_class(cfg.crossatn)

        # use another FC on output before pooling
        self.output_fc = None
        if cfg.use_output_fc:
            self.output_fc = MLP(input_dim, cfg.output_fc_config)
            # correct current input dim
            input_dim = cfg.output_fc_config.output_dim

        # build pooler
        self.pooler = make_pooler_module(input_dim, cfg.pooler_config.name, cfg.pooler_config)

        # correct current input dim, depending on pooler
        if cfg.pooler_config.name == PoolerConst.ATN:
            if cfg.pooler_config.num_layers > 1:
                input_dim *= cfg.pooler_config.num_layers
        self.output_dim = input_dim

        self.action_head = nn.Linear(self.output_dim, ACTION_HEAD)
        # run the initializer
        init_network(self, cfg.weight_init_type, cfg.weight_init_std)
        self.cfg = cfg

    def calculate_output_size(self) -> int:
        """
        Calculate output feature dim of this transformer model

        Returns:
            Output feature dim.
        """
        output_dim = self.output_dim
        if self.use_context:
            output_dim += self.cfg.crossatn.hidden_dim
        return output_dim

    def forward(self, features: th.FloatTensor, mask: th.BoolTensor, lengths: th.LongTensor, hidden_state: Optional[th.FloatTensor]):
        """
        COOT forward pass. This is used in RetrievalModelManager to compute the embeddings.

        Args:
            features: Input features with shape (batch_size, max_seq_len, dim_features)
            mask: Mask with 0 for real data, 1 for padded elements to be ignored. Shape (batch_size, max_seq_len)
            lengths: Sequence length per datapoint (must correspond to the mask) shape (batch_size)
            hidden_state: Optional hidden state for cross-attention with shape (batch_size, dim_hidden)

        Returns:
            Tuple of:
                Features after pooling with shape (batch_size, dim_output)
                Features before pooling with shape (batch_size, max_seq_len, dim_hidden)
        """
        # print("ATN IN: feat",features.shape,"mask",mask.shape)
        # (batch, seq, input_dim)

        # dropout input
        if self.input_dropout is not None:
            features = self.input_dropout(features)

        # normalize input
        if self.norm_input is not None:
            features = self.norm_input(features)

        # convert input with FC
        if self.input_fc is not None:
            features = self.input_fc(features)
            # print("input fc",features.shape)
            # (batch, seq, new_dim)

        # add CLS token from global or local encoder
        if self.net_cls is not None:
            features, mask, _length = self.net_cls(features, mask, lengths)

        # add temporal encoding
        if self.embedding is not None:
            features = self.embedding(features)

        # apply transformer
        features = self.tf(features, features, features, mask)
        # print("after transformer", features.shape, len(atns), atns[0].shape)

        # (batch, seq, new_dim)

        # apply transformer context
        add_after_pool = None
        if self.use_context:
            assert hidden_state is not None
            # hidden state shape (batch, dim_clip)

            # context as query, features as key, value
            # output will be size 1 (add after pooling)

            hidden_state = cast(th.FloatTensor, hidden_state.unsqueeze(1))

            # here we have to actually mask the query
            mask_ctx = mask

            ctx = self.tf_context(hidden_state, features, features, mask_ctx)
            # output is size 1, remove the dim for pooling
            # print(f"context out {ctx.shape}")

            add_after_pool = ctx.squeeze(1)

        # apply pooling
        pooled = self.pooler(features, mask, lengths)

        if add_after_pool is not None:
            # concatenate result of global context cross-attention to global representation
            pooled = th.cat([pooled, add_after_pool], dim=-1)

        # print("after pooler", pooled.shape)
        # (batch, new_dim)

        # apply output FC
        if self.output_fc is not None:
            pooled = self.output_fc(pooled)

        action_logits = self.action_head(pooled)
        actions_preds = nn.Sigmoid()(action_logits)
        # if self.linear_out:
        #     pooled = self.linear_out(pooled)
        return pooled, features, actions_preds


class LearnableClsToken(nn.Module):
    """
    Layer that adds learnable CLS tokens to sequence input.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # initialization will be handled by model
        cls_token = th.zeros(d_model)
        self.cls_param = nn.Parameter(cls_token, requires_grad=True)

        # define a single 1 as unlearnable parameter here, so that it will
        # automatically be transfered to GPU
        self.fixed_ones = nn.Parameter(th.ones(1), requires_grad=False)

    def forward(self, features, mask, lengths):
        """
        CLS Token forward.
        """
        # features shape (batch, seq_len, model_dim)
        # mask shape (batch, seq_len)
        # lengths shape (batch)
        # print(f"CLS mean/std {self.cls_param.mean():.9f}, "
        #       f"{self.cls_param.std():.9f}")
        batch, _seq_len, _d_model = features.shape
        # add cls token to features
        features = th.cat([self.cls_param.unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1), features], dim=1)
        assert th.all(features[0, 0, :] == self.cls_param)
        # add Falses to beginning of the mask
        zeros = (self.fixed_ones.unsqueeze(0).repeat(batch, 1) * 0).bool()  # shape (batch, 1)
        mask = th.cat([zeros, mask], dim=1)
        # increment all lengths by one. not inplace!
        # noinspection PyAugmentAssignment
        lengths = lengths + 1
        return features, mask, lengths


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder.
    """

    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()

        self.cfg = cfg
        assert self.cfg.num_layers > 0, f"{self.cfg.num_layers} layers in transformer is invalid"
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(
                self.cfg.hidden_dim, self.cfg.num_heads, self.cfg.pointwise_ff_dim, self.cfg.dropout,
                self.cfg.activation.name, activation_cfg=self.cfg.activation, norm_name=self.cfg.norm.name,
                norm_cfg=self.cfg.norm)
                for _ in range(self.cfg.num_layers)])

    def forward(self, query, key, value, mask):
        """
        Args:
            query: (batch_size, query_len, d_model)
            key: (batch_size, key_len, d_model)
            value: (batch_size, key_len, d_model)
            mask: (batch_size, key_len)

        Returns:
            output (batch_size, query_len, d_model)
        """
        batch_size, query_len, _embed_dim = query.shape
        batch_size, key_len, _embed_dim = key.shape
        # for this transformer architecture, mask needs to be expanded

        # NOT squared instead of expanded:
        # having entire rows be zero means the softmax will be uniform.
        # the query will never attend to masked keys, and useless masked query
        # parts can be discarded on the output pool

        mask_expanded = mask.unsqueeze(1).expand(batch_size, query_len, key_len)
        # print(mask_expanded.shape)
        # (batch_size, query_len, key_len) dtype bool

        sources = query
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(query, key, value, mask_expanded)
            query = key = value = sources
        return sources


class TransformerEncoderLayer(nn.Module):
    """
    Self Attention Layer as in BERT.
    """

    def __init__(
            self, d_model: int, num_heads: int, d_ff: int, dropout_prob: float = 0.,
            activation_name: str = ActivationConst.GELU, activation_cfg: Optional[ActivationConfig] = None,
            norm_name: str = NormalizationConst.LAYERNORM_COOT,
            norm_cfg: Optional[NormalizationConfig] = None):
        super().__init__()

        if d_ff == 0:
            d_ff = d_model
        self.self_attention_layer = Sublayer(
            MultiHeadAttention(num_heads, d_model, dropout_prob),
            d_model, norm_name=norm_name, norm_cfg=norm_cfg)

        self.pointwise_feedforward_layer = Sublayer(
            PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob, activation_name, activation_cfg=activation_cfg),
            d_model, norm_name=norm_name, norm_cfg=norm_cfg)
        
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, sources_mask):
        """
        Args:
            query: (batch_size, query_len, d_model)
            key: (batch_size, key_len, d_model)
            value: (batch_size, key_len, d_model)
            sources_mask: (batch_size, query_len, key_len)

        Returns:
            output: (batch_size, query_len, d_model)
        """
        # sources: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)

        sources = self.self_attention_layer(query, key, value, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)

        return sources


class Sublayer(nn.Module):
    """
    Add Residual and Layernorm to the given layer.
    """

    def __init__(self, sublayer, d_model: int, norm_name: str, norm_cfg: Optional[NormalizationConfig]):
        super().__init__()

        self.sublayer = sublayer
        self.layer_normalization = make_normalization_module(d_model, norm_name, norm_cfg)

    def forward(self, *args):
        """
        Sublayer forward.
        """
        # save input for residual
        x = args[0]
        # run sublayer
        sublayer_return = self.sublayer(*args)
        if isinstance(sublayer_return, th.Tensor):
            # sublayer returns only a tensor
            x = sublayer_return + x
            return self.layer_normalization(x)
        # sublayer returns other information, too
        x = sublayer_return[0] + x
        return self.layer_normalization(x)#, *sublayer_return[1:]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention.
    """

    def __init__(self, num_heads, d_model, dropout_prob):
        super().__init__()

        assert d_model % num_heads == 0,\
            f"model dim {d_model} not divisible by {num_heads} heads"

        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.query_projection = nn.Linear(d_model, num_heads * self.d_head)
        self.key_projection = nn.Linear(d_model, num_heads * self.d_head)
        self.value_projection = nn.Linear(d_model, num_heads * self.d_head)
        self.final_projection = nn.Linear(d_model, num_heads * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)

        self.attention = None

    def forward(self, query, key, value, mask_expanded: Optional[th.BoolTensor] = None, _layer_cache=None):
        """
        value_len must be equal to key_len
        query_len is the output length

        Args:
            query: (batch_size, query_len, model_dim)
            key: (batch_size, key_len, model_dim)
            value: (batch_size, key_len_len, model_dim)
            mask_expanded: (batch_size, query_len, key_len)
            _layer_cache: DecoderState (unused)

        Returns:
            output: (batch_size, query_len, model_dim)
        """
        # print("attention mask", mask)
        batch_size, query_len, d_model = query.size()

        d_head = d_model // self.num_heads

        query_projected = self.query_projection(query)  # shape (batch_size, query_len, num_heads, d_head)
        key_projected = self.key_projection(key)  # shape (batch_size, key_len, num_heads, d_head)
        value_projected = self.value_projection(value)  # shape (batch_size, key_len, num_heads, d_head)

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        query_heads = query_projected.view(batch_size, query_len, self.num_heads, d_head).transpose(1, 2)
        # print("query_heads", query_heads.shape)
        # (batch_size, num_heads, query_len, d_head)

        key_heads = key_projected.view(batch_size, key_len, self.num_heads, d_head).transpose(1, 2)
        # print("key_heads", key_heads.shape)
        # (batch_size, num_heads, key_len, d_head)

        value_heads = value_projected.view(batch_size, value_len, self.num_heads, d_head).transpose(1, 2)
        # print("value_heads", value_heads.shape)
        # (batch_size, num_heads, key_len, d_head)

        attention_weights = self.scaled_dot_product(query_heads, key_heads)
        # print("attention_weights", attention_weights.shape)
        # (batch_size, num_heads, query_len, key_len)

        if mask_expanded is not None:
            mask_expanded_per_head = mask_expanded.unsqueeze(1).expand_as(attention_weights)
            # print("mask_expanded_per_head", mask_expanded_per_head.shape)
            # shape (batch_size, num_heads, query_len, key_len)
            attention_weights = attention_weights.masked_fill(mask_expanded_per_head, -nntrainer.typext.INF)
            # print("attention_weights", attention_weights.shape)
            # shape (batch_size, num_heads, query_len, query_len)

        # DONT Save attention to the object
        attention = self.softmax(attention_weights)
        # print("attention_weights", attention_weights.shape)

        attention_dropped = self.dropout(attention)
        context_heads = th.matmul(attention_dropped, value_heads)
        # shape (batch_size, num_heads, query_len, d_head)
        # print("context_heads", context_heads.shape)

        context_sequence = context_heads.transpose(1, 2)
        # (batch_size, query_len, num_heads, d_head)

        context = context_sequence.reshape(batch_size, query_len, d_model)
        # (batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        # print("final_output", final_output.shape)

        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        """
        Args:
             query_heads: (batch_size, num_heads, query_len, d_head)
             key_heads: (batch_size, num_heads, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = th.matmul(query_heads, key_heads_transposed)
        # (batch_size, num_heads, query_len, key_len)

        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):
    """
    Feedforward on last dimension (pointwise) with default activation Relu
    and DropOut.
    """

    def __init__(self, d_ff, d_model, dropout_prob, activation_name: str = "gelu",
                 activation_cfg: Optional[ActivationConfig] = None):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            make_activation_module(activation_name, activation_cfg),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        """
        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)
