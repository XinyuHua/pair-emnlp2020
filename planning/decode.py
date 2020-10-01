import time
import os
import json
import torch

from strategies import DecodingStrategy
from task import TextPlanningTask
from modules.checkpoint_utils import load_checkpoint_to_cpu
from modules.progress_bar import build_progress_bar
import utils
from options import get_decode_parser


def main():
    start_time = time.time()
    parser = get_decode_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    task = TextPlanningTask.setup_task(args, is_inference=True)
    task.load_dataset(args.test_set)

    strategy = DecodingStrategy(args, task_dict=task.dictionary)

    args.ckpt_dir = f'../checkpoints/planning/{args.domain}/{args.exp_name}/checkpoint_best.pt'
    assert os.path.exists(args.ckpt_dir), f"{args.ckpt_dir} path does not exist!"

    state = load_checkpoint_to_cpu(path=args.ckpt_dir)
    model = task.build_model(args)
    model.load_state_dict(state['model'], strict=True)
    model = model.cuda()
    model.eval()

    itr = task.get_batch_iterator(dataset=task.dataset(args.test_set),
                                  max_tokens=args.max_tokens,
                                  max_samples=args.max_samples,
                                  max_positions=args.max_positions,
                                  seed=args.seed,
                                  epoch=0).next_epoch_itr(shuffle=False)

    results = []
    progress_bar = build_progress_bar(args, itr)
    translations = generate_batched_itr(progress_bar, strategy, model,
                                        task.task_dict)

    fout = open(f'output/{args.exp_name}_{args.domain}_{args.test_set}.jsonl', 'w')

    for trans in translations:
        cur_obj = dict()
        cur_obj['id'] = trans['id']

        prompt_string = task.dictionary.decode(trans['prompt_ids'])
        prompt_toks = task.dictionary.convert_ids_to_tokens(trans['prompt_ids'])
        cur_obj['prompt_string'] = prompt_string
        cur_obj['prompt_tokens'] = prompt_toks

        kp_src_string = task.dictionary.decode(trans['kp_src_ids'])
        kp_src_toks = task.dictionary.convert_ids_to_tokens(trans['kp_src_ids'])
        cur_obj['kp_src_str'] = kp_src_string
        cur_obj['kp_src_toks'] = kp_src_toks

        kp_tgt_string = task.dictionary.decode(trans['kp_tgt_ids'])
        kp_tgt_toks = task.dictionary.convert_ids_to_tokens(trans['kp_tgt_ids'])
        cur_obj['kp_tgt_str'] = kp_tgt_string
        cur_obj['kp_tgt_toks'] = kp_tgt_toks

        offset = trans['offset']
        template, modified_offset = create_template(kp_tgt_toks, offset, kp_src_toks)
        cur_obj['template'] = template
        cur_obj['modified_offset'] = modified_offset
        cur_obj['offset'] = offset

        if not args.quiet:
            print('ID-{}: OP: {}'.format(cur_obj['id'], prompt_string))
            print('KP-src: {}'.format(kp_src_string))
            print('KP-tgt: {}'.format(kp_tgt_string))
            print()

        results.append(cur_obj)
        fout.write(json.dumps(cur_obj) + '\n')

    fout.close()

    time_elapsed = time.time() - start_time
    print("decoding finished in {:.2f} seconds".format(time_elapsed))
    print("{:.2f} seconds per example".format(time_elapsed / len(results)))


def generate_batched_itr(data_itr, strategy, model, task_dict):

    for sample in data_itr:
        s = utils.move_to_cuda(sample)

        with torch.no_grad():

            hypos, kp_offset_pred = strategy.generate(model, s)

            for batch in range(hypos.size(0)):
                example_id = s['id'][batch]
                src_ids = s['net_input']['input_ids'][batch].tolist()

                ret_obj = {'id': example_id}

                gtruth_kp_tgt = s['kp_target'][batch].tolist()
                ref_kp_tgt_len = s['kp_target_length'][batch].item()
                gtruth_kp_tgt = gtruth_kp_tgt[:ref_kp_tgt_len]
                ret_obj['gtruth_kp_tgt'] = gtruth_kp_tgt

                prompt_end = src_ids.index(task_dict.sep())
                ret_obj['prompt_ids'] = src_ids[:prompt_end]


                kp_src_end = src_ids.index(task_dict.pad()) if task_dict.pad() in src_ids else len(src_ids)
                kp_src_ids = src_ids[prompt_end + 1: kp_src_end]
                kp_tgt_ids = None
                ret_obj['kp_tgt_ids'] = kp_tgt_ids
                ret_obj['kp_src_ids'] = kp_src_ids

                hypo = hypos[batch].tolist()


                if task_dict.bok() in hypo:
                    kp_tgt_start = hypo.index(task_dict.bok())
                    kp_tgt_end = hypo.index(task_dict.eos()) if task_dict.eos() in hypo else len(hypo)
                    kp_tgt_ids = hypo[kp_tgt_start + 1: kp_tgt_end]
                    ret_obj['kp_tgt_ids'] = kp_tgt_ids

                # if task_dict.bos() in hypo:
                #     hypo_start = hypo.index(task_dict.bos())
                #     hypo_end = hypo.index(task_dict.eos())
                #     hypo = hypo[hypo_start:]
                #
                #     generated_tgt = hypo[: hypo_end]
                #     ret_obj['tgt'] = generated_tgt

                cur_kp_offset_pred = kp_offset_pred[batch]
                ret_obj['offset'] = cur_kp_offset_pred

                yield ret_obj


def create_template(kp_tgt, offset, kp_src_toks=None):
    """Based on the offset prediction, create template with KP-plan.

    When offset is already done on first word, no need to consult kp_src_toks.
    Otherwise, check to ensure no break of keyphrase.

    Args:
        kp_tgt (list[str]): such as ['a', 'good', 'post', '[SEP]', 'anyone', '##s', 'views',
                                     '[SEP]', '[SEP]']
        offset (list[int]): such as [3, 4, 8, 8, 2, 7, 8, 9, 3]
        kp_src_toks (list[str]): such as ['a', 'good', 'post', '[SEP]', 'position', '[SEP]',
                                     'anyone', '##s', 'views', '[SEP]', 'San', '##e', 'people',
                                     '[SEP]', 'Everyone', '[SEP]']

    Returns:
        template (list[str]): template tokens
    """
    base = 0
    template = []

    if kp_src_toks is not None:
        # modify offset to ensure no break of keyphrase
        new_offset = mask_non_first_word(kp_src_toks, kp_tgt)
        new_offset = [o if new_o == 1 else 0 for o, new_o in zip(offset, new_offset)]
        offset = new_offset


    for k, o in zip(kp_tgt, offset):
        if o == 0:
            template.append(k)
        else:
            cur_pos = base + o
            while len(template) < cur_pos - 1:
                template.append(None)
            template.append(k)
            if k == '[SEP]':
                base += o
    return template, offset


def mask_non_first_word(kp_src_toks, kp_tgt):
    """Create a list of binary indicators for first word of keyphrases."""
    offset = []
    possible_kps = set()
    cur_kp = []
    for w in kp_src_toks:
        if w == '[SEP]':
            possible_kps.add(tuple(cur_kp))
            cur_kp = []
        else:
            cur_kp.append(w)

    first2kp = dict()
    for kp in possible_kps:
        f = kp[0]
        if f not in first2kp:
            first2kp[f] = []
        first2kp[f].append(kp)

    ptr = 0
    while ptr < len(kp_tgt):
        progressed = False
        cur_tok = kp_tgt[ptr]
        if cur_tok == '[SEP]':
            offset.append(1)
            progressed = True
            ptr += 1

        elif cur_tok in first2kp:
            kp_lst = first2kp[cur_tok]
            for k in kp_lst:
                if len(k) + ptr < len(kp_tgt) \
                    and k == tuple(kp_tgt[ptr: ptr + len(k)]):
                    offset.append(1)
                    offset.extend([0] * (len(k) - 1))
                    ptr += len(k)
                    progressed = True
                    break

            if not progressed:

                for i in range(len(kp_tgt), ptr + 1, -1):
                    cand_seg = kp_tgt[ptr: i]
                    if '[SEP]' in cand_seg: continue

                    for kp in possible_kps:
                        kp_str = ' '.join(kp)
                        cand_str = ' '.join(cand_seg)

                        if cand_str in kp_str:
                            offset.append(1)
                            offset.extend([0] * (i - ptr - 1))
                            ptr += len(cand_seg)
                            progressed = True
                            break

                    if progressed:
                        break

        if not progressed:
            ptr += 1
            offset.append(1)

    assert len(offset) == len(kp_tgt), '{} != {}'.format(len(offset), len(kp_tgt))
    return offset




if __name__=='__main__':
    main()
