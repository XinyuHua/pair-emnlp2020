"""Refinement"""
import json
import torch
from tqdm import tqdm
from dataset import BaseDataset
import utils

class RefinementDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = f'../data/refinement/{self.domain}/{self.set_type}.jsonl'
        self.template = []
        self.load_raw_dataset(path=path)

    def load_raw_dataset(self, path):
        """Load raw data for refinement dataset"""

        print(f'loading {path}')
        for ln in tqdm(open(path)):
            cur_obj = json.loads(ln)

            prompt_str = cur_obj["prompt"]
            prompt_ids = self.tokenizer.encode(prompt_str,
                                               max_length=self.max_prompt_len,
                                               truncation=True,
                                               add_special_tokens=False)


            if 'kp_tgt_str' in cur_obj:
                appended_src = ' <s> ' + cur_obj['kp_tgt_str'].replace('<mask>', '<s>')
            else:
                appended_src = ' <s> ' + cur_obj['kp_plan_str']

            appended_ids = self.tokenizer.encode(appended_src,
                                                 max_length=self.max_kp_len,
                                                 truncation=True,
                                                 add_special_tokens=False)

            cur_src_ids = prompt_ids + appended_ids
            self.source.append(cur_src_ids)
            self.ID.append(cur_obj['id'])

            # whenever <unk> is present, reconstruct text first and then retokenize
            cur_template = cur_obj['template']
            cur_template = [w if (w is not None and w != '[SEP]') else '<mask>'
                            for w in cur_template]
            cur_template_ids = self.tokenizer.convert_tokens_to_ids(cur_template)
            if 3 in cur_template_ids:
                # <unk> encountered, needs to reconstruct the text
                temp_str = ''
                for w in cur_template:
                    if w == '<mask>':
                        temp_str += ' ' + w
                    elif w == 'Ġ.':
                        temp_str += '.'
                    elif w[0] == 'Ġ':
                        temp_str += ' ' + w[1:]
                    else:
                        temp_str += w
                temp_str = temp_str.strip()
                if temp_str[0] == 'Ġ':
                    temp_str = temp_str[1:]
                cur_template_ids = self.tokenizer.encode(temp_str, add_speical_tokens=False)

            cur_template_ids = cur_template_ids[:self.max_tgt_len]
            self.template.append(cur_template_ids + [self.mask_idx])

            if self.is_inference:
                kp_set = cur_obj['kp_set_str']
                kp_set_list = [' ' + w.strip() for w in kp_set.split('<s>')]
                kp_set_ids = [self.tokenizer.encode(cur_kp, add_special_tokens=False)
                              for cur_kp in kp_set_list]
                self.kp_set.append(kp_set_ids)


            if self.is_inference and self.use_system_plan: continue

            tgt_ids = self.tokenizer.encode(cur_obj['tgt'],
                                            max_length=self.max_tgt_len,
                                            truncation=True,
                                            add_special_tokens=False)
            tgt_ids = [self.bos_idx] + tgt_ids + [self.eos_idx]
            self.target.append(tgt_ids)



    def __getitem__(self, index):
        """
        setup `PAIR-light`:
            source: src_ids + <s> + [random-mask] (train)
                    src_ids + <s> + [all-mask] (inference)
            target: tgt_ids

        setup `PAIR-full`:
            source: src_ids + <s> + [non-kp-mask/mix-mask] (train)
                    src_ids + <s> + [non-kp-mask] (inference)
            target: tgt_ids
        """
        cur_id = self.ID[index]
        src_ids = self.source[index]
        src_len = len(src_ids)

        template_ids = self.template[index]
        template_ids = torch.LongTensor(template_ids)

        if not self.is_inference or not self.use_system_plan:
            tgt_ids = self.target[index]

        def generate_mask(strategy):
            num_masks = self.random.randint(1, len(tgt_ids) - 2)
            ind = self.random.choice(len(tgt_ids) - 2, size=num_masks, replace=False)

            if strategy == 'random':
                draft = torch.LongTensor(tgt_ids)
                draft[ind + 1] = self.mask_idx

            elif strategy == 'non-kp-random':
                masked_draft = [t if t not in ind else self.mask_idx for t in tgt_ids]
                non_kp_masked_draft = []
                for i in range(1, len(masked_draft)):
                    if template_ids[i - 1] != self.mask_idx:
                        non_kp_masked_draft.append(template_ids[i - 1])
                    else:
                        non_kp_masked_draft.append(masked_draft[i])
                draft = torch.LongTensor(template_ids)

            return draft


        if self.is_inference:

            # in setup 26 we supply the draft with all masks, using the
            # average length of training set
            if self.setup == 'pair-light':
                if self.domain == 'arggen':
                    draft_len = 120
                elif self.domain == 'opinion':
                    draft_len = 223
                else:
                    draft_len = 315

                draft = torch.ones(draft_len + 1).long() * self.mask_idx

            else:
                # in setup `pair-full`, use non-kp as initial draft
                draft = torch.LongTensor(template_ids)


            if not self.use_system_plan:
                target = torch.LongTensor(tgt_ids)
            else:
                target = None
        else:
            if self.setup == 'pair-light':
                draft = generate_mask('random')

            else:
                # training `pair-full`, using a mixture of random mask
                # and non-kp random mask
                strategy_option = self.random.uniform(0, 1)
                if strategy_option > 0.5:
                    strategy_option = 'random'
                else:
                    strategy_option = 'non-kp-random'
                draft = generate_mask(strategy_option)

            target = torch.LongTensor(tgt_ids)

        encoder_input = torch.cat([torch.LongTensor(src_ids),
                                   torch.LongTensor(self.bos_idx),
                                   draft[:-1]])

        ret_obj = dict(
                id=cur_id,
                encoder_input=encoder_input,
                src_len=src_len
        )

        if target is not None:
            ret_obj['target'] = target
        if self.is_inference:
            ret_obj['kp_set'] = self.kp_set[index]
            ret_obj['template'] = template_ids
        return ret_obj