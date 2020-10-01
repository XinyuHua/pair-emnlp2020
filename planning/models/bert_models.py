import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from .bert_blocks import (
    BertEmbeddings,
    BertTransformer,
    BertPredictionHead,
    BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BertLayerNorm
)


class BertModelForPlanning(nn.Module):
    """The main model will predict tokens (kp-tgt and tgt). And the auxiliary
    model predicts keyphrase offset as a sequence tagging task.

    The main model has a mixture of bidirectional self-attention (op and kp-src)
    and causal self-attention (kp-tgt and tgt). The auxiliary model has
    bidirectional self-attention only (kp-offset). The weights of auxiliary
    model is tied to the main model except the last layer for tagging.

    We consider offset prediction with possible range [0, 128].
    """

    def __init__(self, config, dictionary, is_inference=False):
        super().__init__()
        self.config = config
        self.config.inference = is_inference
        self.add_bert_config()

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertTransformer(config)
        self.task_dictionary = dictionary
        self.padding_idx = self.task_dictionary.pad_index
        self.mask_idx = self.task_dictionary.mask_index
        self.cls = BertPredictionHead(config)

        self.load_pretrained_bert(model_name=config.model_name)
        self.max_positions = self.config.max_position_embeddings

        self.num_possible_word_offsets = 128
        config.num_possible_word_offsets = 128
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.offset_tagger = nn.Linear(config.hidden_size,
                                       self.num_possible_word_offsets)
        self.apply(self._init_weights)

    @classmethod
    def build_model(cls, args, task, is_inference=False):
        task_dict = task.task_dict
        return cls(args, task_dict, is_inference)


    def add_bert_config(self):
        bert_config_path = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP[self.config.model_name]
        with open(bert_config_path) as fin:
            bert_config = json.load(fin)

        for k in bert_config:
            setattr(self.config, k, bert_config[k])
        self.config.layer_norm_eps = 1e-12
        self.config.output_attentions = False
        self.config.output_hidden_states = False


    def load_pretrained_bert(self, model_name):
        """Load pre-trained weights of BERT"""
        model_path = BERT_PRETRAINED_MODEL_ARCHIVE_MAP[model_name]
        old_state = torch.load(model_path, map_location='cpu')

        def correct_old_param_naming(old_name):
            if 'bert' in old_name:
                old_name = old_name[5:]

            if 'gamma' in old_name:
                return old_name.replace('gamma', 'weight')

            if 'beta' in old_name:
                return old_name.replace('beta', 'bias')

            return old_name

        state = {correct_old_param_naming(k) : v for k, v in old_state.items()}
        missing_keys = []
        unexpected_keys = []

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module, prefix=""):
            local_metadata = {}
            module._load_from_state_dict(
                state, prefix, local_metadata, True, missing_keys, unexpected_keys, []
            )

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self, prefix='')
        self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def predict_kp_offset(self, input_ids, position_ids):
        """Predict KP-offset:

        Args:
            input_ids: padded kp-plan word ids
        """
        padding_mask = input_ids.eq(0)
        token_type_ids = torch.ones(input_ids.shape).long().cuda()
        past = [None] * len(self.encoder.layer)
        attention_mask = padding_mask[:, None, None, :].to(dtype=next(self.parameters()).dtype)
        attention_mask = attention_mask * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, past_length=-1,
            position_ids=position_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past=past,
        )
        kp_enc_out = self.dropout(encoder_outputs[0])
        wo_output = self.offset_tagger(kp_enc_out) * input_ids.ne(0).unsqueeze(-1)
        return wo_output


    def get_normalized_probs(self, encoder_out, log_probs):
        """Get normalized probabilities (or log probs) from a net's output"""
        # encoder_out = #net_output['transformer_out']
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError


    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                kp_tgt_mask=None,
                position_ids=None,
                labels=None,
                kp_offset_ids=None,
                past=None):

        """Run forward pass, return LM output and word offset output.
        """

        input_shape = input_ids.size()
        device = input_ids.device

        if past is None:
            past_length = 0
            past = [None] * len(self.encoder.layer)
        else:
            past_length = past[0][0].size(-2)

        if attention_mask is None:
            attention_mask = input_ids.eq(self.padding_idx)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            batch_size, seq_length = input_shape

            def build_mask_by_token_type_ids():
                """Build hybrid-causal mask for attention base on token-type-ids"""
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) > seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3

                # unmask the source input (op + kp)
                token_type_ids_ = token_type_ids.eq(1)[:, None, :].repeat(1, seq_length, 1)
                # token_type_ids_1 = token_type_ids_.detach().cpu().numpy()
                # causal_mask_ = causal_mask.long().detach().cpu().numpy()
                causal_mask = causal_mask & token_type_ids_
                # causal_mask_2 = causal_mask.long().detach().cpu().numpy()

                return causal_mask[:, None, :, :] | attention_mask[:, None, None, :]

            if self.config.inference:
                extended_attention_mask = attention_mask[:, None, None, :]

            else:
                # training time, build mask by token-type-ids
                extended_attention_mask = build_mask_by_token_type_ids()

        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        if attention_mask is not None:
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            extended_attention_mask = extended_attention_mask * -10000.0
        else:
            extended_attention_mask = None

        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, past_length=past_length,
            position_ids=position_ids,
        )
        # ea = extended_attention_mask.detach().cpu().numpy()
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            past=past,
        )

        # sequence_output: (batch, src_len, dim)
        # sequence_output = encoder_outputs[0].transpose(0, 1)
        sequence_output = encoder_outputs[0]
        sequence_output = self.dropout(sequence_output)
        lm_output = self.cls(sequence_output)

        wo_output = self.offset_tagger(sequence_output)
        # during training time, select the ones where kp_tgt_mask=1
        if not self.config.inference:
            wo_output_ = []
            max_kp_tgt_len = max(kp_tgt_mask.sum(dim=1))
            for batch in range(batch_size):
                cur_mask = kp_tgt_mask[batch].bool() # [len]
                cur_pred = wo_output[batch][cur_mask] # [len x 128]
                padding = torch.zeros((max_kp_tgt_len - len(cur_pred), 128),
                                      dtype=cur_pred.dtype).cuda()
                # cur_pred = torch.cat((cur_pred, padding.half()))
                cur_pred = torch.cat((cur_pred, padding))
                wo_output_.append(cur_pred.unsqueeze(0))

            wo_output = torch.cat(wo_output_)


        return {'lm_out': lm_output,
                'transformer_padding_mask': attention_mask,
                'encoder_states': encoder_outputs,
                'offset_out': wo_output}
