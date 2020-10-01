import time
import json
import os
import logging

from tqdm import tqdm
import torch
import torch.nn as nn

from system import BARTSeq2seq
from options import get_generation_parser
import utils

from decoding_strategy import (
    SinglePassDecoding,
    MultiPassDecoding,
)

logger = logging.getLogger(__name__)

MASK_TOK = '<mask>'
BOS_TOK = '<s>'
BOS_IDX = 0
PAD_IDX = 1
EOS_IDX = 2
MASK_IDX = 50264

def generate():
    parser = get_generation_parser()
    args = parser.parse_args()

    ckpt_path = utils.get_latest_ckpt_path(args.ckpt_dir)
    print(f'Evaluating on {ckpt_path}')

    t0 = time.time()
    checkpoint = torch.load(ckpt_path)
    print('Checkpoint loading time: {:.2f} secs'
          .format(time.time() - t0))

    if not os.path.exists(f'output/{args.domain}/'):
        os.makedirs(f'output/{args.domain}/')
    output_path = f'output/{args.domain}/{args.output_name}.jsonl'

    system = BARTSeq2seq(hparams=args, is_inference=True).cuda()
    test_dataloader = system.test_dataloader()
    system.load_state_dict(checkpoint['state_dict'])
    if args.fp16:
        system.model = system.model.half()

    if args.n_gpus > 1:
        system.model = nn.DataParallel(system.model)

    fout = open(output_path, 'w')

    if args.setup in ['seq2seq', 'kpseq2seq']:
        # never enforce, max_tgt_len is the size of training set
        decoding_strategy = SinglePassDecoding(args, system.tokenizer)
    else:
        decoding_strategy = MultiPassDecoding(args, system.tokenizer)

    for batch in tqdm(test_dataloader):

        net_inputs = utils.move_to_cuda(batch['net_input'])
        generation_results = decoding_strategy.generate(system.model,
                                                        batch)

        if args.setup in ['seq2seq', 'kpseq2seq']:
            output_ids, output_probs = generation_results
        else:
            output_ids, refinement_history, masking_history, prob_history, \
               external_prob_history, sampling_history, sampling_selection_history, \
               token_force_history = generation_results

        for b, sample_id in enumerate(batch['id']):

            hypo_ids = output_ids[b]
            hypo_ids = hypo_ids[hypo_ids.ne(2) & hypo_ids.ne(1) & hypo_ids.ne(0)]
            hypo_toks = system.tokenizer.convert_ids_to_tokens(hypo_ids)
            hypo_len = len(hypo_ids)
            hypo_str = system.tokenizer.decode(hypo_ids, skip_special_tokens=True).strip()

            src_ids = net_inputs['input_ids'][b]
            src_ids = src_ids[src_ids.ne(1)] # remove padding
            src_toks = system.tokenizer.convert_ids_to_tokens(src_ids)

            if args.setup in ['b1', 'b2', '24']: # no draft in src
                if MASK_TOK in src_toks:
                    op_toks = src_toks[:src_toks.index(MASK_TOK)]
                elif BOS_TOK in src_toks:
                    op_toks = src_toks[:src_toks.index(BOS_TOK)]
                else:
                    op_toks = src_toks
            else:
                op_toks = src_toks[:src_toks.index(BOS_TOK)]

            ret_obj = dict(
                id=sample_id,
                output_str=hypo_str,
                output_toks=hypo_toks,
                op_toks=op_toks,
                src_toks=src_toks,
            )

            if 'lm_label' in batch:
                cur_tgt_ids = batch['lm_labels'][b]
                cur_tgt_ids = cur_tgt_ids[cur_tgt_ids >= 0]
                cur_tgt_str = system.tokenizer.decode(cur_tgt_ids, skip_special_tokens=True).strip()
                ret_obj['gtruth_tgt'] = cur_tgt_str

            if args.setup in ['b1', 'b2', '24']:
                fout.write(json.dumps(ret_obj) + "\n")
                continue


            # 1. decoded_ids for each refinement iteration
            # 2. model probs for each refinement, external prob history
            # 3. masking history
            # 4. forcing history
            # 5. optional: sampling history
            cur_refine = []
            cur_model_probs = []
            cur_masking_history = []
            cur_force_history = []

            # stats only exist for sampling decoding
            if sampling_selection_history is not None:
                cur_sampling_history = []
                cur_external_probs = []
                cur_sampling_selection_history = []

            for itr, ref in enumerate(refinement_history[b]):
                ref_wids = [wid for wid in ref if wid != PAD_IDX]
                cur_refine.append(system.tokenizer.convert_ids_to_tokens(ref_wids))

                cur_ref_len = len(ref_wids)
                cur_model_prob = [-1] + prob_history[b][itr]
                cur_model_prob = cur_model_prob[:cur_ref_len]
                cur_model_probs.append(cur_model_prob)

                cur_masking_history.append(system.tokenizer.convert_ids_to_tokens(masking_history[b][itr]))
                cur_force_history.append(token_force_history[b][itr])

                if sampling_selection_history is not None:
                    cur_external_prob = [-1, -1] + external_prob_history[b][itr]
                    cur_external_prob = cur_external_prob[:cur_ref_len]
                    cur_external_probs.append(cur_external_prob)
                    cur_sampling_selection_history.append(sampling_selection_history[b][itr])

                    cur_itr_sampling_history = []
                    for samp in sampling_history[b][itr]:
                        samp_words = samp[1]
                        samp_nll = samp[0]
                        samp = [wid for wid in samp_words if wid != PAD_IDX]
                        samp = system.tokenizer.convert_ids_to_tokens(samp)
                        cur_itr_sampling_history.append((samp_nll, samp))

                    cur_sampling_history.append(cur_itr_sampling_history)


            last_seq_model = [-1] + prob_history[b][-1]
            last_seq_model = last_seq_model[:hypo_len]
            cur_model_probs.append(last_seq_model)

            ret_obj['refinement_history'] = cur_refine
            ret_obj['model_probs_history'] = cur_model_probs
            ret_obj['masking_history'] = cur_masking_history
            ret_obj['force_history'] = cur_force_history

            if sampling_selection_history is not None:
                cur_itr_sampling_history = []
                for samp in sampling_history[b][-1]:
                    samp_words = samp[1]
                    samp_nll = samp[0]
                    samp = [wid for wid in samp_words if wid != PAD_IDX]
                    samp = system.tokenizer.convert_ids_to_tokens(samp)
                    cur_itr_sampling_history.append((samp_nll, samp))

                cur_sampling_history.append(cur_itr_sampling_history)

                last_seq_external = [-1, -1] + external_prob_history[b][-1]
                last_seq_external = last_seq_external[:hypo_len]
                cur_external_probs.append(last_seq_external)
                cur_sampling_selection_history.append(sampling_selection_history[b][-1])
                ret_obj['sampling_selection_history'] = cur_sampling_selection_history
                ret_obj['external_lm_probs_history'] = cur_external_probs
                ret_obj['sampling_history'] = cur_sampling_history

            fout.write(json.dumps(ret_obj) + "\n")

    fout.close()

if __name__ == '__main__':
    generate()