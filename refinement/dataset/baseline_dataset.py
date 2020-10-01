"""BART baseline dataset that has no drafts, for seq2seq and kpseq2seq"""
import json
import torch
from tqdm import tqdm
from dataset import BaseDataset
import utils

class BaselineDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = f'../data/{self.domain}/refinement_{self.set_type}.jsonl'
        self.load_raw_dataset(path=path)


    def load_raw_dataset(self, path):
        """Load raw data for baseline models."""

        print(f'loading {path}')
        for ln in tqdm(open(path)):
            cur_obj = json.loads(ln)



            prompt_str = cur_obj['prompt']
            prompt_ids = self.tokenizer.encode(prompt_str,
                                               max_length=self.max_prompt_len,
                                               truncation=True,
                                               add_special_tokens=False)

            if self.setup == 'seq2seq':
                cur_src_ids = prompt_ids
            elif self.setup == 'kpseq2seq':
                appended_src = ' <mask> ' + cur_obj['kp_set_str']
                appended_ids = self.tokenizer.encode(appended_src,
                                                     truncation=True,
                                                     max_length=self.max_kp_len,
                                                     add_special_tokens=False)
                cur_src_ids = prompt_ids + appended_ids


            self.source.append(cur_src_ids)
            self.ID.append(cur_obj['id'])
            if self.is_inference:
                kp_set = cur_obj['kp_set_str']
                kp_set_list = [' ' + w.strip() for w in kp_set.split('<s>')]
                kp_set_ids = [self.tokenizer.encode(cur_kp, add_special_tokens=False)
                              for cur_kp in kp_set_list]
                self.kp_set.append(kp_set_ids)

            if self.is_inference and self.use_system_plan:
                continue

            tgt_ids = self.tokenizer.encode(cur_obj['tgt'],
                                            truncation=True,
                                            max_length=self.max_tgt_len,
                                            add_special_tokens=False)
            tgt_ids = [self.bos_idx] + tgt_ids + [self.eos_idx]
            self.target.append(tgt_ids)



    def __getitem__(self, index):
        cur_id = self.ID[index]
        src_ids = self.source[index]
        enc_input = torch.LongTensor(src_ids)

        ret_obj = dict(
            id=cur_id,
            encoder_input=enc_input,
        )

        if not self.is_inference or not self.use_system_plan:
            tgt_ids = self.target[index]
            tgt_ids = torch.LongTensor(tgt_ids)
            ret_obj['target'] = tgt_ids

        if self.is_inference:
            ret_obj['kp_set'] = self.kp_set[index]

        return ret_obj


    def collater(self, samples):
        def merge(key, is_list=False):
            if is_list:
                res = []
                for i in range(len(samples[0][key])):
                    res.append(utils.collate_tokens(
                        [s[key][i] for s in samples], pad_idx=1,
                    ))
                return res
            else:
                return utils.collate_tokens(
                    [s[key] for s in samples], pad_idx=1,
                )

        input_ids = merge('encoder_input')
        attn_mask = input_ids.ne(1).long()
        net_input = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )

        ret_obj = dict(
            id=[s['id'] for s in samples],
        )

        if "target" in samples[0]:
            y = merge('target')
            dec_in = y[:, :-1].contiguous()
            dec_in[dec_in == self.eos_idx] = self.pad_idx

            lm_labels = y[:, 1:].clone()
            lm_labels[lm_labels == self.pad_idx] = -100
            # tgt = [bos w1 w2 ... wn eos]
            # dec_in = [bos, w1, w2, ... wn]
            # lm_labels = [w1, w2, ..., wn, eos]
            net_input['decoder_input_ids'] = dec_in
            ret_obj['lm_labels'] = lm_labels

        ret_obj['net_input'] = net_input
        if 'kp_set' in samples[0]:
            ret_obj['kp_set'] = [s['kp_set'] for s in samples]

        return ret_obj
