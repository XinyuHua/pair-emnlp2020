import random
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import activations
from models.configuration import BartConfig
from models.modeling_utils import (
    LayerNorm,
    create_position_ids_from_input_ids,
)

class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = activations.ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x, attn_weights = self.self_attn(query=x, key=x, key_padding_mask=encoder_padding_mask,)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))

        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn_weights


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx,)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim)


    def forward(
        self, input_ids, attention_mask=None,
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *self.output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            assert attention_mask.dim() == 2
            attention_mask = attention_mask.eq(0)


        inputs_embeds = self.embed_tokens(input_ids)
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if self.output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)

            if self.output_attentions:
                all_attentions.append(attn)

        if self.output_hidden_states:
            encoder_states.append(x)

        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        return x, encoder_states, all_attentions


class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = activations.ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self, x, encoder_hidden_states, encoder_attn_mask=None, layer_state=None, attention_mask=None,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        # next line mutates layer state
        x, self_attn_weights = self.self_attn(query=x, key=x, layer_state=layer_state, attn_mask=attention_mask,)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key

        x, encoder_attn_weights = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.output_past = config.output_past
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_tokens = embed_tokens
        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model)

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        combined_mask,
        decoder_cached_states=None,
        generation_mode=False,
        **unused
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        # check attention mask and invert
        if encoder_padding_mask is not None:
            assert encoder_padding_mask.dim() == 2
            encoder_padding_mask = encoder_padding_mask.eq(0)

        # embed positions
        positions = self.embed_positions(input_ids, generation_mode=generation_mode)

        if generation_mode:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids)
        x += positions
        contains_nan = (x!=x).any()

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)  # (seq_len, BS, model_dim)
        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []

        for i, decoder_layer in enumerate(self.layers):
            decoder_layer  # type: DecoderLayer
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[i] if decoder_cached_states is not None else None
            x, layer_self_attn, layer_past = decoder_layer(
                x, encoder_hidden_states, encoder_padding_mask, layer_state=layer_state, attention_mask=combined_mask,
            )

            if self.output_past:
                next_decoder_cache.append(layer_past.copy())
            if self.output_hidden_states:
                all_hidden_states += (x,)
            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert shapes from (seq_len, BS, model_dim) to (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)

        if self.output_past:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # if attn_mask is not None:
        #     am = attn_mask.cpu().detach().numpy()
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv = self.encoder_decoder_attention  # type: bool
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.masked_fill(reshaped, -10000.)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training,)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask = saved_state.get("prev_key_padding_mask", None)  # type: Optional[Tensor]
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
        self, input_dim, inner_dim, num_classes, pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int,
    ):
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        assert padding_idx is not None
        num_embeddings += padding_idx + 1  # WHY?
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input, generation_mode=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if generation_mode:  # the position is our current step in the decoded sequence
            pos = int(self.padding_idx + input.size(1))
            positions = input.data.new(1, 1).fill_(pos)
        else:
            positions = create_position_ids_from_input_ids(input, self.padding_idx)
        return super().forward(positions)

