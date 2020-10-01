import torch
import torch.nn.functional as F
import utils
import strategies.strategy_utils as strategy_utils
import numpy as np


class DecodingStrategy:

    def __init__(self, args, task_dict):
        super().__init__()
        self.args = args
        self.task_dict = task_dict

        self.bos = task_dict.bos()
        self.sep = task_dict.sep()
        self.bok = task_dict.bok()
        self.pad = task_dict.pad()
        self.eos = task_dict.eos()

        self.use_ref_length = args.use_ref_length
        self.predict_keyphrase_offset = args.predict_keyphrase_offset
        self.force_kp_copy = args.force_kp_copy
        self.kp_gen_max_time = args.kp_gen_max_time
        self.quiet = args.quiet
        self.do_sampling = args.do_sampling
        self.repetition_penalty = args.repetition_penalty
        self.disable_ngram_repeat = args.disable_ngram_repeat
        self.predict_keyphrase_offset = args.predict_keyphrase_offset
        self.max_gen_tgt_length = args.max_gen_tgt_length


    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Create inputs for BERT model. If past is not None, only
        take the last token as input_ids for model input."""
        inputs = dict()
        if 'past' in kwargs and kwargs['past']:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            inputs['token_type_ids'] = input_ids.new(input_ids.shape).fill_(1)
        inputs['input_ids'] = input_ids
        inputs.update(kwargs)
        return inputs


    def generate(self, model, sample):
        """Generate output from left to right.

        seq: prompt [SEP] kp-set [BOK] kp-tgt [EOS]

        Args:
            sample:
                - `net_input`: (batch, max_src_len)
                - `token_type_ids`: same dimension as net_input with all 0s
        """

        net_input = sample['net_input']
        batch_size = net_input['input_ids'].shape[0]

        shortest_src = min(sample['src_length']).item()
        max_gen_steps = 510 - shortest_src

        kp_offsets_pred = [[] for _ in range(batch_size)]

        # indicate beginning of decoding
        start_off_tok_id = self.bok
        bok_ids = torch.empty((batch_size, 1)).fill_(start_off_tok_id).long().cuda()
        input_ids = torch.cat((net_input['input_ids'],  bok_ids), dim=-1)

        token_ones = torch.ones((batch_size, 1), dtype=torch.long).cuda()
        token_type_ids = torch.cat((net_input['token_type_ids'], token_ones), dim=-1)

        self.maybe_print_src_info(input_ids)

        if self.force_kp_copy:
            allowed_kp_wids = strategy_utils.get_allowed_kp_wids(
                                        input_ids=net_input['input_ids'],
                                        eog=self.eos)

        if self.kp_gen_max_time > 0:
            # each keyphrase can be used at most this many times
            kp_used_freq, kp_unique_words = strategy_utils.setup_kp_stats(
                                                    net_input['input_ids'])

        # mask out paddings
        attention_masks = input_ids.eq(self.pad).float().cuda()

        # position_ids will jump over the padding tokens
        position_ids = torch.LongTensor([
            list(range(input_ids.shape[1])) for i in range(input_ids.shape[0])
        ]).cuda()
        padding_cnt = position_ids.shape[1] - sample['src_length'] - 1
        for ix in range(batch_size):
            position_ids[ix, -1] = position_ids[ix, -1] - padding_cnt[ix]

        unfinished_sents = input_ids.new(batch_size).fill_(1).unsqueeze(1)
        kp_prediction_unfinished = input_ids.new(batch_size).fill_(1).unsqueeze(1)

        past = None
        cur_step = 0
        cur_tgt_step = input_ids.new(batch_size).fill_(0).unsqueeze(1)
        cur_kp_tgt_step = input_ids.new(batch_size).fill_(0).unsqueeze(1)

        # generate until the last sequence can reach max_gen_steps, for other
        # sequences that reach it or reach 511, set unfinished_sents[batch]=0
        # and use 511 as position embedding to avoid error in BERT embeddings
        while cur_step < max_gen_steps:
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              past=past,
                                                              attention_mask=attention_masks,
                                                              position_ids=position_ids,
                                                              )

            if 'token_type_ids' not in model_inputs: # first step
                model_inputs['token_type_ids'] = token_type_ids

            model_inputs = utils.move_to_cuda(model_inputs)
            outputs = model(**model_inputs)
            past = outputs['encoder_states'][1]
            next_token_logits = outputs['lm_out'][:, -1, :].unsqueeze(1)

            disabled_ids = [set() for _ in range(batch_size)]
            if self.force_kp_copy:
                disabled_ids = strategy_utils.enforce_copy(disabled_ids,
                                                           allowed_kp_wids,
                                                           kp_prediction_unfinished,
                                                           vocab_size=next_token_logits.shape[-1])

            if self.kp_gen_max_time > 0:
                disabled_ids = strategy_utils.enforce_kp_times(disabled_ids,
                                                      kp_used_freq,
                                                      kp_unique_words,
                                                      kp_prediction_unfinished,
                                                      self.kp_gen_max_time)

            if self.disable_ngram_repeat > 0:
                # disable repetition of ngram
                disabled_ids = strategy_utils.enforce_non_ngram_repeat(disabled_ids, input_ids, self.disable_ngram_repeat)

            for batch in range(batch_size):
                next_token_logits[batch][0][list(disabled_ids[batch])] = -10000.

            if self.repetition_penalty > 1.0:
                # apply repetition penalty to punish same prediction as
                # previous token.
                # note: This will not apply if the sequence is finished.
                prev_tokens = input_ids[:, -1]
                for i in range(input_ids.shape[0]):
                    if next_token_logits[i, 0, prev_tokens[i]] < 0:
                        next_token_logits[i, 0, prev_tokens[i]] *= self.repetition_penalty
                    else:
                        next_token_logits[i, 0, prev_tokens[i]] /= self.repetition_penalty

            if self.do_sampling:
                # run sampling based decoding
                if self.args.temperature != 1.0:
                    next_token_logits = next_token_logits / self.args.temperature
                next_token_logits = strategy_utils.top_k_top_p_filtering(next_token_logits.squeeze(1),
                                                          top_k=self.args.sampling_topk,
                                                          top_p=self.args.sampling_topp)

                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            else:
                # run greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            if cur_step > 0 and self.predict_keyphrase_offset:
                cur_offset_pred = torch.argmax(outputs['offset_out'], dim=-1)
                for i in range(batch_size):
                    if kp_prediction_unfinished[i] > 0:
                        kp_offsets_pred[i].append(cur_offset_pred[i][-1].item())


            tokens_to_add = next_token * kp_prediction_unfinished + self.pad * (1 - kp_prediction_unfinished)
            kp_prediction_unfinished.mul_(tokens_to_add.ne(self.eos))
            input_ids = torch.cat([input_ids, tokens_to_add], dim=-1)

            self.maybe_print_token_info(next_token_logits, cur_step, tokens_to_add)

            attention_masks = torch.cat([attention_masks, torch.zeros((attention_masks.shape[0], 1)).float().cuda()], dim=1)

            position_ids = (position_ids[:, -1] + 1).unsqueeze(-1)
            position_ids[position_ids > 511] = 511
            # position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

            cur_step += 1
            for batch in range(batch_size):
                if kp_prediction_unfinished[batch] == 0:
                    cur_tgt_step[batch] += 1
                else:
                    cur_kp_tgt_step[batch] += 1

            unfinished_sents.mul_(tokens_to_add.ne(self.eos))
            if max(unfinished_sents) == 0:
                break

            if self.kp_gen_max_time > 0:
                kp_used_freq = strategy_utils.update_kp_used_freq(kp_used_freq,
                                                         input_ids,
                                                         kp_prediction_unfinished)

        return input_ids, kp_offsets_pred


    def predict_offset(self, model, batch_size, input_ids, kp_offsets_pred):
        if kp_offsets_pred is None:
            return
        kp_plan_padded, kp_position_ids = self.get_kp_plan_from_input_ids(input_ids)
        wo_output = model.predict_kp_offset(input_ids=kp_plan_padded,
                                            position_ids=kp_position_ids)
        wo_pad_mask = torch.any(wo_output.ne(0.0), dim=-1)
        wo_probs = model.get_normalized_probs(wo_output, log_probs=False)
        pred_wo_probs, pred_offset_ids = torch.max(wo_probs, dim=-1)

        for batch in range(batch_size):
            cur_pred = pred_offset_ids[batch]
            cur_pad_mask = wo_pad_mask[batch].sum().item()
            kp_offsets_pred[batch] = cur_pred[:cur_pad_mask].tolist()



    def get_kp_plan_from_input_ids(self, input_ids):
        """Base on BOK and BOS, get kp-plan ids and pad them"""
        result_lst = []
        position_ids_lst = []

        for batch in range(len(input_ids)):
            cur_lst = input_ids[batch].tolist()
            bok_pos = cur_lst.index(self.bok)
            bos_pos = cur_lst.index(self.bos)

            cur_plan = cur_lst[bok_pos + 1: bos_pos]
            cur_pos_id_start = input_ids[batch][:bok_pos].ne(0).sum().item() + 1
            position_ids_lst.append(list(range(cur_pos_id_start,
                                           cur_pos_id_start + len(cur_plan))))

            result_lst.append(cur_plan)

        max_kp_len = max([len(x) for x in result_lst])
        position_ids = np.zeros([len(input_ids), max_kp_len])
        results = np.zeros([len(input_ids), max_kp_len])
        for batch, item in enumerate(result_lst):
            results[batch][:len(item)] = item
            cur_pos_ids = position_ids_lst[batch]
            position_ids[batch][:len(cur_pos_ids)] = cur_pos_ids
        results = torch.LongTensor(results)
        position_ids = torch.LongTensor(position_ids)
        return results.cuda(), position_ids.cuda()

    def maybe_print_token_info(self, next_token_logits, cur_step, tokens_to_add):
        if self.quiet:
            return

        real_probs = F.softmax(next_token_logits, dim=-1)
        topk_val, topk_ind = torch.topk(real_probs, k=3)

        topk_ind = topk_ind.squeeze(1)
        topk_val = topk_val.squeeze(1).detach().cpu().numpy()

        topk_output_str = []
        for i in range(topk_ind.shape[0]):
            topk_output_str_ = ''
            topk_str = self.task_dict.convert_ids_to_tokens(topk_ind[i])
            for ix, item in enumerate(topk_str):
                cur_output_str = '{}({:.3f})'.format(item, topk_val[i][ix])
                topk_output_str_ += '{:<20} '.format(cur_output_str)

            topk_output_str.append(topk_output_str_)

        tokens_to_add_str_list = self.task_dict.convert_ids_to_tokens(tokens_to_add)
        tokens_to_add_ = tokens_to_add.squeeze().tolist()

        tokens_to_add_str = ''
        for x, xid in zip(tokens_to_add_str_list, tokens_to_add_):
            cur_str = '{} ({})'.format(x, xid)
            tokens_to_add_str += '{:15}\t'.format(cur_str)

        print('step-{:<3}\ttokens_to_add: {}\n' \
              .format(cur_step,
                      tokens_to_add_str,
                      ))
        for item in topk_output_str:
            print(item)
        print('-' * 50)


    def maybe_print_src_info(self, input_ids):
        if self.quiet:
            return
        src_kp_sep = strategy_utils.extract_src_and_kp_from_input_ids(input_ids,
                                                                      self.task_dict)
        for batch, rst in enumerate(src_kp_sep):
            print('sample-{}, src: {}'.format(batch, ' '.join(rst['src'])))
            kp_str = '; '.join([' '.join(kp) for kp in rst['kp']])
            print('kp: {}'.format(kp_str))
