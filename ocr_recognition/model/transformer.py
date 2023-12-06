"""Based on https://github.com/opconty/Transformer_STR"""
import math
import copy
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_


def _get_clones(module: nn.Module, num: int) -> nn.ModuleList:
    """Returns a ModuleList containing deep copies of the given module.

    Args:
        module (nn.Module): The module to be cloned.
        num (int): Number of clones.

    Returns:
        nn.ModuleList: ModuleList containing cloned modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def _get_activation_fn(activation: str) -> nn.functional:
    """Returns the activation function based on the provided string.

    Args:
        activation (str): String representing the activation function ("relu" or "gelu").

    Returns:
        nn.functional: Activation function.

    Raises:
        RuntimeError: Raises when the activation name is incorrect.
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class TransformerEncoder(nn.Module):
    """TransformerEncoder implementation PyTorch."""

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None
    ):
        """TransformerEncoder constructor.

        Args:
            encoder_layer (nn.Module): An instance of the transformer encoder layer.
            num_layers (int): The number of encoder layers in the transformer.
            norm (nn.Module, optional): Normalization layer. Defaults to None.
        """
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.tensor,
        mask: Optional[torch.tensor] = None,
        src_key_padding_mask: Optional[torch.tensor] = None
    ) -> torch.tensor:
        """Pass the input through the encoder layers in turn.

        Tensor shapes may find in Transformer class docs.

        Args:
            src (torch.tensor): The sequence to the encoder (required).
            mask (torch.tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.tensor, optional): Mask for the src keys per batch.

        Returns:
            torch.tensor: Forward result.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MultiheadAttention(nn.Module):
    """Multihead self-attention mechanism."""

    __annotations__ = {
        "bias_k": torch._jit_internal.Optional[torch.Tensor],
        "bias_v": torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = ["q_proj_weight", "k_proj_weight",
                     "v_proj_weight", "in_proj_weight"]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None
    ):
        """
        Multi-head self-attention constructor.

        Args:
            embed_dim (int): Dimension of input feature embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout. Default is 0.
            bias (bool): Should learnable bias be added to the attention weights. Default is True.
            add_bias_kv (bool): Should bias be added to the kv projections. Default is False.
            add_zero_attn (bool): Should adds a vector for masked positions. Default is False.
            kdim (int, optional): Dimension of the key projections. Defaults to embed_dim.
            vdim (int, optional): Dimension of the value projections. Defaults to embed_dim.

        Raises:
            AttributeError: Raises when embed_dim don't divisible by num_heads.
        """
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise AttributeError("embed_dim must be divisible by num_heads")

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Resets parameters of MultiHead module."""
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        """Support loading old MultiheadAttention.

        Checkpoints generated by v1.1.0.

        Args:
            state: The state dictionary containing module parameters.
        """
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True
        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: torch.tensor,
        key: torch.tensor,
        value: torch.tensor,
        key_padding_mask: Optional[torch.tensor] = None,
        need_weights: Optional[bool] = True,
        attn_mask: Optional[torch.tensor] = None
    ) -> torch.tensor:
        """Forward method.

        Args:
            query (torch.tensor): Query tensor.
            key (torch.tensor): Key tensor.
            value (torch.tensor): Value tensor.
            key_padding_mask (torch.tensor, optional): Mask tensor indicating positions with
                padding values. Defaults to None.
            need_weights (bool. optional): Should returns the attention weights. Default is True.
            attn_mask (torch.tensor, optional): Mask tensor for preventing attention to
                certain positions. Defaults to None.

        Returns:
            torch.tensor: Output tensor after applying the transformer encoder layer.
        """
        if not self._qkv_same_embed_dim:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight
            )
        else:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask
            )


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer implementation."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """Single layer of the transformer encoder constructor.

        Args:
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of feedforward layer. Defaults to 2048.
            dropout (float): Dropout probability. Defaults to 0.1.
            activation (str): Activation function used in feedforward layer. Defaults to "relu".
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Sets the state of the transformer encoder layer.

        Args:
            state (dict of str: Any): The state dictionary containing module parameters.
        """
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: torch.tensor,
        src_mask: torch.tensor = None,
        src_key_padding_mask: torch.tensor = None
    ) -> torch.tensor:
        """Forward method.

        Args:
            src (torch.tensor): Source input tensor.
            src_mask (torch.tensor, optional): Mask tensor for preventing attention to
                certain positions. Defaults to None.
            src_key_padding_mask (torch.tensor, optional): Mask tensor indicating positions with
                padding values. Defaults to None.

        Returns:
            torch.tensor: Output tensor after applying the transformer encoder layer.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(
            self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class PositionalEncoding(nn.Module):
    """Implementation of positional encoding module."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Positional encoding module constructor.

        Args:
            d_model (int): Dimension of the model.
            dropout (float): Dropout probability. Defaults to 0.1.
            max_len (int): Maximum length of the input sequence. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, data: torch.tensor) -> torch.tensor:
        """Forward method.

        Args:
            data (torch.tensor): Data to froward.

        Returns:
            torch.tensor: Data after forward.
        """
        data = data + self.pe[:data.size(0), :]
        return self.dropout(data)


class Transformer(nn.Module):
    """Create Transformer model."""

    def __init__(self, num_class: int, config: Dict[str, Any]):
        """Transformer constructor.

        Args:
            num_class (int): Amount of chars in vocabulary.
            config (Dict[str, Any]): Transformer params. You may set in
                configs/model/transformer.yaml.
        """
        super(Transformer, self).__init__()

        num_head: int = config["num_head"]
        num_layer: int = config["num_layer"]
        hidden_size: int = config["hidden_size"]
        dropout: float = config["dropout"]
        self.inp_channel: int = config["inp_channel"]

        encoder_layers: nn.Module = TransformerEncoderLayer(
            self.inp_channel, num_head, hidden_size, dropout)

        self.src_mask: Optional[torch.Tensor] = None
        self.pos_encoder: nn.Module = PositionalEncoding(self.inp_channel, dropout)
        self.transformer_encoder: nn.Module = TransformerEncoder(encoder_layers, num_layer)
        self.decoder: nn.Module = nn.Linear(self.inp_channel, num_class)
        self.init_weights()

    def _generate_square_subsequent_mask(self, matrix_size: int) -> torch.tensor:
        """Generates a square subsequent mask for the transformer.

        Args:
            matrix_size (int): Size of the square matrix.

        Returns:
            torch.tensor: Square subsequent mask tensor.
        """
        mask = (torch.triu(torch.ones(matrix_size, matrix_size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(
            mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self) -> None:
        """Initializes the weights of the model."""
        initrange: float = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.tensor) -> torch.tensor:
        """Forward method.

        Args:
            src (torch.tensor): Data to froward.

        Returns:
            torch.tensor: Data after forward.
        """
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
            self.src_mask = mask

        src = src * math.sqrt(self.inp_channel)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
