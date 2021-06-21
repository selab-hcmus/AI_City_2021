from typing import Any, Dict, List, Optional, cast

from nntrainer.models.mlp import MLPConfig
from nntrainer.typext import ConfigClass, ConstantHolder
from nntrainer.models.poolers import PoolerConfig
from nntrainer.models.activations import ActivationConfig
from nntrainer.models.normalizations import NormalizationConfig

class TransformerConfig(ConfigClass):
    """
    Configuration class for a single coot network

    Args:
        config: Configuration dictionary to be loaded, dataset part.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.name: str = config.pop("name")
        self.output_dim: int = config.pop("output_dim")  # output dim must be specified for future modules in the chain
        self.dropout_input: float = config.pop("dropout_input")
        self.norm_input: str = config.pop("norm_input")
        self.positional_encoding: str = config.pop("positional_encoding")

        # Add learnable CLS token as first element to the input sequences
        self.add_local_cls_token: bool = config.pop("add_local_cls_token")
        if self.add_local_cls_token:
            self.local_cls_token_init_type: str = config.pop("local_cls_token_init_type")
            self.local_cls_token_init_std: float = config.pop("local_cls_token_init_std")

        # Add input FC to downsample input features to the transformer dimension
        self.use_input_fc: bool = config.pop("use_input_fc")
        if self.use_input_fc:
            self.input_fc_config = MLPConfig(config.pop("input_fc_config"))

        # Self-attention
        self.selfatn = None
        field_selfatn = "selfatn_config"
        if self.selfatn is None:
            self.selfatn = TransformerEncoderConfig(config.pop("selfatn_config"))

        # output FC for resampling features before pooling
        self.use_output_fc: bool = config.pop("use_output_fc")
        if self.use_output_fc:
            self.output_fc_config = MLPConfig(config.pop("output_fc_config"))

        # cross-attention
        self.use_context: bool = config.pop("use_context")
        if self.use_context:
            # fields required for cross-attention
            field_crossatn = "crossatn_config"
            config_class = TransformerEncoderConfig
            self.crossatn = config_class(config.pop(field_crossatn))
        # pooler
        self.pooler_config = PoolerConfig(config.pop("pooler_config"))

        # weight initialiazion
        self.weight_init_type: str = config.pop("weight_init_type")
        self.weight_init_std: float = config.pop("weight_init_std")

        self.linear_out: bool = config.pop("linear_out", False)


class TransformerEncoderConfig(ConfigClass):
    """
    TransformerEncoder Submodule

    Args:
        config: Configuration dictionary to be loaded.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # load fields required for a transformer
        self.hidden_dim: int = config.pop("hidden_dim")
        self.num_layers: int = config.pop("num_layers")
        self.dropout: float = config.pop("dropout")
        self.num_heads: int = config.pop("num_heads")
        self.pointwise_ff_dim: int = config.pop("pointwise_ff_dim")
        self.activation = ActivationConfig(config.pop("activation"))
        self.norm = NormalizationConfig(config.pop("norm"))


class TransformerTypesConst(ConstantHolder):
    """
    Store network types for COOT.

    Notes:
        TRANSFORMER_LEGACY: Transformer as used in the paper.
        RNN_LEGACY: CMHSE Paper GRU.
    """
    TRANSFORMER_LEGACY = "transformer"
    TRANSFORMER_TORCH = "transformer_torch"
    RNN_LEGACY = "rnn"
