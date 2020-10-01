"""Modified from Huggingface"""
import random
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from transformers import PreTrainedModel
from models.modeling_utils import (
    _prepare_bart_decoder_inputs,
    _filter_out_falsey_values,
    _make_linear_from_emb,
    _reorder_buffer
)

from models.bart_blocks import *

BART_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/pytorch_model.bin",
    "bart-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-mnli/pytorch_model.bin",
    "bart-large-cnn": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/pytorch_model.bin",
}


class PretrainedBartModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    pretrained_model_archive_map = BART_PRETRAINED_MODEL_ARCHIVE_MAP

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        decoder_input_ids, decoder_attn_mask = _prepare_bart_decoder_inputs(self.config, input_ids,)
        dummy_inputs = {
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_attention_mask": decoder_attn_mask,
        }
        return dummy_inputs


# Public API
class BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()


    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,  # type: Tuple
        decoder_attention_mask=None,
        decoder_cached_states=None,
        generation_mode=False,
    ):

        # make masks if user doesn't supply
        if not generation_mode:
            decoder_input_ids, decoder_attention_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attn_mask=decoder_attention_mask,
                mask_dtype=self.shared.weight.dtype,
            )
        assert decoder_input_ids is not None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        assert isinstance(encoder_outputs, tuple)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            generation_mode=generation_mode,
        )
        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs = _filter_out_falsey_values(decoder_outputs)  # type: tuple
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs = _filter_out_falsey_values(encoder_outputs)  # type: tuple
        return decoder_outputs + encoder_outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


class BartForConditionalGeneration(PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        # if base_model is None:
        base_model = BartModel(config)
        self.model = base_model
        self.lm_head = _make_linear_from_emb(self.model.shared)

    def tie_weights(self):
        pass  # hack to prevent changing lm_head.out_features. The input and output embeddings are still the same.


    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None,
        generation_mode=False,
        **unused
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."
            model = BartForConditionalGeneration.from_pretrained('bart-large')
            input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]
            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)
            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            generation_mode=generation_mode,
        )
        lm_logits = self.lm_head(outputs[0])
        outputs = (lm_logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in lm_labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs[:2]

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "generation_mode": True,
        }

    def prepare_scores_for_generation(self, scores, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(scores, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(scores, self.config.eos_token_id)
        return scores

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            # reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
            # reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
            reordered_past.append(layer_past_new)
        new_enc_out = enc_out if enc_out is None else enc_out.index_select(1, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return self.lm_head
