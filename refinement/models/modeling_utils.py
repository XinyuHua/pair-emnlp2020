import torch
import torch.nn as nn
import numpy as np

LARGE_NEGATIVE = -1e4

def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache

def _prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_attn_mask=None, mask_dtype=None,
):
    """Prepare masks that ignore padding tokens in the decoder and a causal lm mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    need_causal_mask = not config.output_past
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()[:2]
    if decoder_attn_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
        if need_causal_mask:
            # causal_lm_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len).type(mask_dtype)), 1)
            causal_lm_mask = torch.triu(torch.FloatTensor(LARGE_NEGATIVE * np.ones((tgt_len, tgt_len))).type(mask_dtype), 1)
        else:
            causal_lm_mask = None
        new_shape = (bsz, tgt_len, tgt_len)
        # make it broadcastable so can just be added to the attention coefficients
        decoder_attn_mask = _combine_masks(decoder_padding_mask, causal_lm_mask, new_shape).to(device=input_ids.device)
        if mask_dtype is not None:
            decoder_attn_mask = decoder_attn_mask.to(mask_dtype)
    assert decoder_attn_mask is None or decoder_attn_mask.shape == (bsz, 1, tgt_len, tgt_len)
    return decoder_input_ids, decoder_attn_mask

def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))


def _combine_masks(key_padding_mask, causal_lm_mask, targ_size):
    """Make one mask of shape (bsz, 1, tgt_len, src_len) """
    a = torch.zeros(targ_size)  # targ_size is(bsz, tgt_len, src_len)
    b = torch.zeros(targ_size)
    if key_padding_mask is not None:  # (bsz, tgt_len) -> targ_size
        _check_shapes(key_padding_mask.shape, targ_size[:2])
        reshaped = key_padding_mask.unsqueeze(2).expand(*targ_size)
        a[reshaped] = LARGE_NEGATIVE

    if causal_lm_mask is not None:  # (tgt_len, src_len) -> targ_size
        _check_shapes(causal_lm_mask.shape, targ_size[-2:])
        b = causal_lm_mask.unsqueeze(0).expand(*targ_size)
    return (a + b).unsqueeze(1).clamp(LARGE_NEGATIVE,)


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))


def _combine_masks(key_padding_mask, causal_lm_mask, targ_size):
    """Make one mask of shape (bsz, 1, tgt_len, src_len) """
    a = torch.zeros(targ_size)  # targ_size is(bsz, tgt_len, src_len)
    b = torch.zeros(targ_size)
    if key_padding_mask is not None:  # (bsz, tgt_len) -> targ_size
        _check_shapes(key_padding_mask.shape, targ_size[:2])
        reshaped = key_padding_mask.unsqueeze(2).expand(*targ_size)
        a[reshaped] = LARGE_NEGATIVE

    if causal_lm_mask is not None:  # (tgt_len, src_len) -> targ_size
        _check_shapes(causal_lm_mask.shape, targ_size[-2:])
        b = causal_lm_mask.unsqueeze(0).expand(*targ_size)
    return (a + b).unsqueeze(1).clamp(LARGE_NEGATIVE, )

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _filter_out_falsey_values(tup):
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indicies = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indicies.long() + padding_idx

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)