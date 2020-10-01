import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .strategy_utils import (
    top_k_top_p_filtering,
    assign_single_value_long,
    tag_kp_tokens_in_paragraph
)
import utils

from decoding_strategy import BaseDecoding

class MultiPassDecoding(BaseDecoding):

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

        self.iterations = args.iterations
        self.setup = args.setup
        assert args.setup in ['pair-light', 'pair-full']

        self.enforce_template_strategy = args.enforce_template_strategy
        self.exempt_p = args.exempt_p

        if self.setup == 'pair-light':
            self.masking_strategy = 'worst-k'
            assert args.enforce_template_strategy == 'none'

        elif self.setup in ['pair-full']:
            self.masking_strategy = 'non-kp-worst-k'
            self.low_kp_prob_threshold = args.low_kp_prob_threshold

        self.sample_times = 1
        if self.do_sampling:
            self.external_lm = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
            self.external_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.external_tokenizer_pad_idx = 50256
            self.sample_times = args.sample_times



    def calculate_nll(self, decoding_results):
        dec_toks = [self.tokenizer.convert_ids_to_tokens(ln.tolist()[1:]) for ln in decoding_results]
        gpt2_ids = [self.external_tokenizer.convert_tokens_to_ids(ln) for ln in dec_toks]
        gpt2_inputs = torch.LongTensor(gpt2_ids).cuda()
        gpt2_output = self.external_lm(gpt2_inputs)

        batch_size = gpt2_inputs.shape[0]

        lm_logits = gpt2_output[0]
        shifted_logits = lm_logits[..., :-1, :].contiguous()
        shifted_labels = gpt2_inputs[..., 1:].contiguous()

        lprobs = utils.get_normalized_probs(shifted_logits, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        probs = utils.get_normalized_probs(shifted_logits, log_probs=False)
        probs = probs.view(-1, probs.size(-1))

        pad_mask = shifted_labels.eq(self.external_tokenizer_pad_idx)
        shifted_labels = shifted_labels.view(-1, 1)

        nll_loss = -lprobs.gather(dim=-1, index=shifted_labels).view(batch_size, -1)
        nll_loss[pad_mask] = 0.0

        external_probs = probs.gather(dim=-1, index=shifted_labels).view(batch_size, -1)
        external_probs[pad_mask] = 1.0

        # TODO: should we use mean instead of sum?
        nll_per_sample = nll_loss.sum(dim=-1)
        sample_lens = decoding_results.ne(self.pad_idx).sum(-1).float()
        ppl_per_sample = torch.exp(nll_per_sample/sample_lens)

        return ppl_per_sample, external_probs

    def _refine_iteration(self, model, encoder_input, encoder_attn_mask,
                         src_lens, max_tgt_len, template):

        batch_size = len(src_lens)
        effective_batch_size = batch_size * self.sample_times
        expanded_batch_idxs = torch.arange(batch_size).view(-1, 1).repeat(1, self.sample_times).view(-1).cuda()

        # if there's no system-level max-tgt-len constraint, use draft lengths
        # plus extra 10 tokens
        if max_tgt_len == -1:
            target_lens = [cur_in[cur_in != 1].shape[0] for cur_in in encoder_input]
            target_lens = [tl - sl for tl, sl in zip(target_lens, src_lens)]
            max_tgt_len = max(target_lens)

        if template is not None:
            # expand template as well
            template = template.index_select(0, expanded_batch_idxs)
            template_lens = template.ne(1).sum(-1)
        else:
            template = None
            template_lens = None

        encoder = model.get_encoder()
        encoder_input_ids = encoder_input
        encoder_attn_mask = encoder_attn_mask
        encoder_outputs = encoder(encoder_input_ids, encoder_attn_mask)
        enc_ = encoder_outputs[0].cpu().detach().numpy()

        dec_input_ids = torch.full(
            (effective_batch_size, 1),
            self.decoder_bos_idx,
            dtype=torch.long,
            device=next(model.parameters()).device
        )

        cur_len = 1
        probs = [[] for _ in range(effective_batch_size)]
        sample_drafts = [[] for _ in range(batch_size)]
        token_force_history = [[] for _ in range(effective_batch_size)]

        encoder_outputs = (encoder_outputs[0].index_select(1, expanded_batch_idxs), *encoder_outputs[1:])
        encoder_attn_mask = encoder_attn_mask.index_select(0, expanded_batch_idxs)
        past = encoder_outputs
        unfinished_seq = dec_input_ids.new(effective_batch_size).fill_(1)

        while cur_len < max_tgt_len:
            model_inputs = self.prepare_inputs_for_generation(
                dec_input_ids, past=past, attention_mask=encoder_attn_mask,
            )

            outputs = model(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            past = outputs[1]

            # when `enforce_kp_strategy` is not `none`, ensure the generation
            # won't stop before it reaches the template length
            if self.enforce_template_strategy in ['force', 'flexible']:
                next_token_logits[cur_len <= template_lens, self.eos_idx] = -10000.

            if self.temperature != 1.0 and self.do_sampling:
                next_token_logits = next_token_logits / self.temperature

            if self.do_sampling:
                next_token_logits = top_k_top_p_filtering(next_token_logits,
                                                          top_k=self.topk,
                                                          top_p=self.topp)
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                chosen_token_probs = next_token_probs.gather(1, next_token)

            else:
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                chosen_token_probs, next_token = torch.max(next_token_probs, dim=-1)
                chosen_token_probs = chosen_token_probs.unsqueeze(-1)


            is_forced = [False for _ in range(effective_batch_size)]
            # enforce KP words from template if the current length is shorter
            # then template
            if self.enforce_template_strategy in ['force', 'flexible'] \
                    and cur_len - 1 < template.shape[1]:

                cur_template = template[:, cur_len - 1]
                no_mask_and_pad = cur_template[cur_template.ne(self.mask_idx)]
                no_mask_and_pad = no_mask_and_pad[no_mask_and_pad.ne(self.pad_idx)]

                # some sample(s) have a non-mask template word needs to be enforced
                if no_mask_and_pad.shape[0] != 0:
                    for b in range(effective_batch_size):
                        cur_template_id = cur_template[b].item()

                        if cur_template_id in [self.pad_idx, self.mask_idx]:
                            continue

                        cur_template_prob = next_token_probs[b, cur_template_id]

                        if self.enforce_template_strategy == 'flexible':
                            # check history (previous 10 tokens)
                            leftmost = max(0, cur_len - 10)
                            _history = dec_input_ids[b][leftmost:].tolist()
                            if cur_template_id in _history:
                                continue

                        chosen_token_probs[b] = cur_template_prob
                        next_token[b] = cur_template[b]
                        is_forced[b] = True

            for b in range(effective_batch_size):
                probs[b].append(chosen_token_probs[b, 0].item())
                token_force_history[b].append(is_forced[b])

            tokens_to_add = next_token.squeeze() * unfinished_seq + (self.pad_idx) * (1 - unfinished_seq)
            dec_input_ids = torch.cat([dec_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            eos_in_seq = (tokens_to_add == self.eos_idx)
            unfinished_seq.mul_((~eos_in_seq).long())

            if not self.quiet:
                output_str = ''
                for b in range(effective_batch_size):
                    w = self.tokenizer.convert_ids_to_tokens([tokens_to_add[b]])[0]
                    p = probs[b][-1]
                    if is_forced[b]:
                        w = f'\033[32m {w} \033[m '
                        output_str += '{:>12}({:.2f})|'.format(w, 100 * p)
                    else:
                        output_str += '{:>12}({:.2f})|'.format(w, 100 * p)
                if cur_len == 1:
                    print('=' * 50)
                print('step={:<3d}|{}'.format(cur_len, output_str))

            if unfinished_seq.max() == 0:
                break

            cur_len += 1

        if not self.do_sampling:
            decoded_rank_selection = dec_input_ids
            probs_chosen = probs
            sample_drafts = None
            external_probs_chosen = None
            selected_indices = None
            token_force_history_sel = token_force_history
        else:
            ppl_per_sample, external_probs = self.calculate_nll(dec_input_ids)
            ppl_per_sample = ppl_per_sample.view(batch_size, -1)

            selected_indices = torch.argmin(ppl_per_sample, dim=-1)
            if not self.quiet:
                for b in range(batch_size):
                    sel = selected_indices[b].item()
                    lens = [len(x[x.ne(self.pad_idx)]) for x in
                            dec_input_ids[b * self.sample_times: (b + 1) * self.sample_times]]
                    print(f'sample-{b}:\n')
                    cur_ppls = ['{:.1f}'.format(p) for p in ppl_per_sample[b].tolist()]
                    print(f'lengths: {lens}\tPPL: {cur_ppls}\tselected: {sel}')

            dec_input_ids_ = dec_input_ids.view(batch_size, self.sample_times, -1)
            max_dec_len = dec_input_ids_.shape[-1]
            decoded_rank_selection = torch.gather(input=dec_input_ids_, dim=1,
                                                  index=selected_indices
                                                  .view(-1, 1, 1)
                                                  .expand(-1, 1, max_dec_len)).squeeze(1)

            probs_chosen = []
            external_probs_chosen = []
            token_force_history_sel = []
            for b in range(batch_size):
                sel = selected_indices[b].item()
                probs_chosen.append(probs[b * self.sample_times + sel])
                external_probs_chosen.append(external_probs[b * self.sample_times + sel].tolist())
                sample_drafts[b] = [(ppl_per_sample[b][j].item(), item.tolist()) for j, item in
                                    enumerate(dec_input_ids_[b])]
                token_force_history_sel.append(token_force_history[b * self.sample_times + sel])

        ret_obj = dict(
            decoded_ids=decoded_rank_selection,
            probs=probs_chosen,
            external_probs=external_probs_chosen,
            drafts=sample_drafts,
            selection_history=selected_indices,
            force_history=token_force_history_sel,
        )
        return ret_obj


    def generate(self, model, batch):
        """Note: if self.enforce_template_strategy is not `none`, the length
        will be forced by the template. Otherwise it will be bounded by
        `max_tgt_len`."""
        net_input = utils.move_to_cuda(batch['net_input'])
        encoder_input_ids = net_input['input_ids']
        encoder_attn_mask = net_input['attention_mask']
        src_lens = batch['src_len']
        batch_size = len(src_lens)

        # pair-light has no template information
        if self.setup == 'pair-light':
            max_tgt_len = self.domain_to_max_len[self.domain]
            sample_template = None
        else:
            max_tgt_len = -1
            sample_template = batch['template'].cuda()

        refinement_history = [[] for _ in range(batch_size)]
        prob_history = [[] for _ in range(batch_size)]
        masking_history = [[] for _ in range(batch_size)]
        token_force_history = [[] for _ in range(batch_size)]


        if self.do_sampling:
            sampling_selection_history = [[] for _ in range(batch_size)]
            external_prob_history = [[] for _ in range(batch_size)]
            sampling_history = [[] for _ in range(batch_size)]
        else:
            sampling_selection_history = None
            external_prob_history = None
            sampling_history = None

        for itr in range(1, self.iterations + 1):
            decoding_results = self._refine_iteration(model=model,
                                                      encoder_input=encoder_input_ids,
                                                      encoder_attn_mask=encoder_attn_mask,
                                                      src_lens=src_lens,
                                                      max_tgt_len=max_tgt_len,
                                                      template=sample_template)

            decoded_ids = decoding_results['decoded_ids']
            model_probs = decoding_results['probs']
            force_history = decoding_results['force_history']
            if self.do_sampling:
                external_probs = decoding_results['external_probs']
                sample_selections = decoding_results['selection_history']
                sample_drafts = decoding_results['drafts']

            for b in range(batch_size):
                prob_history[b].append(model_probs[b])
                token_force_history[b].append(force_history[b])

                if self.do_sampling:
                    sampling_history[b].append(sample_drafts[b])
                    external_prob_history[b].append(external_probs[b])
                    sampling_selection_history[b].append(sample_selections[b].item())


            if itr == self.iterations:
                break

            for b in range(batch_size):
                refinement_history[b].append(decoded_ids[b].tolist())

            dec_lens = decoded_ids.ne(self.pad_idx).sum(-1)
            cur_num_masks = (dec_lens.float() * (1.0 - (itr / self.iterations))).long()

            masked_ind = self.mask_worst_k(decoding_results=decoded_ids,
                                           probs_for_masking=model_probs,
                                           num_mask=cur_num_masks,
                                           kp_set=batch['kp_set'],
                                           spare_kp=('non-kp' in self.masking_strategy),
                                           )
            assign_single_value_long(decoded_ids, masked_ind, self.mask_idx)

            max_dec_len = decoded_ids.shape[1]
            if max_dec_len + max(src_lens) > encoder_input_ids.shape[1]:
                encoder_input_ids_ = torch.ones(batch_size, max_dec_len + max(src_lens)).long().cuda()
                for b in range(batch_size):
                    encoder_input_ids_[b][:src_lens[b]] = encoder_input_ids[b][:src_lens[b]]

                encoder_input_ids = encoder_input_ids_

            for b in range(batch_size):
                cur_dec_len = decoded_ids[b].ne(self.pad_idx).sum()
                cur_mask = decoded_ids[b].clone()
                cur_mask = cur_mask[cur_mask.ne(self.pad_idx)]
                masking_history[b].append(cur_mask.tolist())
                encoder_input_ids[b][src_lens[b]:] = 1
                encoder_input_ids[b][src_lens[b]: src_lens[b] + cur_dec_len] = decoded_ids[b][:cur_dec_len]

            modified_max_input_len = encoder_input_ids.eq(self.pad_idx).sum(dim=-1).min()
            if modified_max_input_len.item() > 0:
                encoder_input_ids = encoder_input_ids[:, :-modified_max_input_len]
            encoder_attn_mask = encoder_input_ids.ne(self.pad_idx).long().cuda()

        return decoded_ids, refinement_history, masking_history, prob_history, \
               external_prob_history, sampling_history, sampling_selection_history, \
               token_force_history


    def mask_worst_k(self, decoding_results, probs_for_masking, num_mask,
                     kp_set=None, spare_kp=False):
        """Mask `num_mask` tokens with the lowest probabilities in `decoding_results`.
        If a token has probability higher than `exempt_p`, it won't be masked.
        If `spare_kp` is set to True, kp tokens (identified by `kp_set`) won't
        be masked, unless it has probability lower than `low_prob_kp_threshold`.

        The following configuration yields vanilla masking (as in cmlm):
            exempt_p = 1.0
            spare_kp = False
        """

        # since the prob can be 1 or 2 off, adjust here
        len_diff = decoding_results.shape[1] - len(probs_for_masking[0])
        probs_for_masking = torch.cat([torch.cat([torch.ones(len_diff), torch.Tensor(item)]).unsqueeze(0)
                                       for item in probs_for_masking], dim=0)

        bsz, seq_len = probs_for_masking.size()
        # first set padding's probs to 1.0 so that they won't be masked
        probs_for_masking[decoding_results.eq(self.pad_idx)] = 1.0

        # get top-k worst
        mask_ind = [probs_for_masking[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1]
                    for batch in range(bsz)]

        # exempt tokens with high probabilities
        if self.exempt_p < 1.0:
            modified_mask_ind = []
            for b in range(bsz):
                cur_masked = mask_ind[b].tolist()
                exempt_proposal = [i for i in cur_masked if probs_for_masking[b][i] >= self.exempt_p]
                if len(exempt_proposal) == len(cur_masked):
                    # HACK: sometimes all masks will be exempted if the model is confident for them
                    # therefore here we first sort them, then guarantee the worst-3 ones will always be masked
                    cur_masked_with_probs = [(i, probs_for_masking[b][i]) for i in cur_masked]
                    sorted_masked = sorted(cur_masked_with_probs, key=lambda x: x[1])

                    cur_modified_mask_ind = [i[0] for i in sorted_masked[:3]]
                else:
                    cur_modified_mask_ind = [i for i in cur_masked if i not in exempt_proposal]
                modified_mask_ind.append(torch.LongTensor(cur_modified_mask_ind).cuda())
            mask_ind = modified_mask_ind

        if spare_kp:
            for b, cur_kp_list in enumerate(kp_set):
                cur_toks = decoding_results[b].tolist()
                kp_tag, kp_start_idx = tag_kp_tokens_in_paragraph(cur_toks, cur_kp_list)

                cur_mask_ind = mask_ind[b].tolist()
                cur_tag_ind = [i for i, tg in enumerate(kp_tag) if tg == 1]
                cur_masked_ind_modified = [k if k not in cur_tag_ind else -1 for k in cur_mask_ind]
                mask_ind_max = max(cur_masked_ind_modified)
                # assert mask_ind_max >= 0, f'after sparing KP, no tokens can be masked! tokens={cur_toks}\n' \
                #                           f'tagged_ind={cur_tag_ind}\ncur_mask_ind={cur_mask_ind}'
                if mask_ind_max < 0:
                    continue

                modified_masked_ind = [k if k >= 0 else mask_ind_max for k in cur_masked_ind_modified]
                mask_ind[b] = torch.LongTensor(modified_masked_ind).cuda()

        mask_ind = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in mask_ind]
        return torch.stack(mask_ind, dim=0)
