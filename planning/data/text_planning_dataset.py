import numpy as np
import torch
import json
import torch.utils.data

from . import data_utils

class TextPlanningDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, dictionary, train=None, seed=None,
                 max_prompt_length=None, max_kp_set_length=None,
                 max_kp_tgt_length=None, inference=False,
                 use_oracle_plan=False):

        self.max_prompt_length = max_prompt_length
        self.max_kp_set_length = max_kp_set_length
        self.max_kp_tgt_length = max_kp_tgt_length
        self.max_positions = 512

        self.prompt = []
        self.kp_set = []
        self.kp_tgt = []
        self.kp_offsets = []
        self.ID = []

        self.lengths = []

        self.train = train
        self.dictionary = dictionary
        self.sep = dictionary.sep()
        self.bos = dictionary.bos()
        self.bok = dictionary.bok()
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()

        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.inference = inference
        self.use_oracle_plan = use_oracle_plan

        self.read_data(data_path)
        self.lengths = np.array(self.lengths)

    def read_data(self, path):
        with open(path) as f:
            for ln in f:
                cur_obj = json.loads(ln)

                def append_token_ids_to_list(field_name, max_len, list_to_append):
                    cur_toks = cur_obj[field_name]
                    if max_len is not None and len(cur_toks) > max_len:
                        cur_toks = cur_toks[:max_len]
                    cur_ids = self.dictionary.convert_tokens_to_ids(cur_toks)
                    list_to_append.append(cur_ids)
                    return len(cur_ids)

                prompt_len = append_token_ids_to_list('prompt_tokens', self.max_prompt_length, self.prompt)
                kp_set_len = append_token_ids_to_list('kp_set', self.max_kp_set_length, self.kp_set)
                kp_tgt_len = append_token_ids_to_list('kp_tgt_tokens', self.max_kp_tgt_length, self.kp_tgt)

                kp_offsets = cur_obj['kp_tgt_offsets']
                if len(kp_offsets) > kp_tgt_len:
                    kp_offsets = kp_offsets[:kp_tgt_len]
                self.kp_offsets.append(kp_offsets)
                self.ID.append(cur_obj["id"])
                self.lengths.append(prompt_len + kp_set_len + kp_tgt_len + 3)
            print('{} samples loaded, average length {:.2f} ({})'.format(
                len(self.lengths), np.mean(self.lengths), path))

    def __len__(self):
        return len(self.prompt)

    def size(self, index):
        return self.lengths[index]

    def num_tokens(self, index):
        return self.lengths[index]

    def __getitem__(self, index):
        ret_obj = self._make_source_target(self.prompt[index],
                                           self.kp_set[index],
                                           self.kp_tgt[index],
                                           self.kp_offsets[index])
        ret_obj['id'] = self.ID[index]
        return ret_obj

    def _make_source_target(self, prompt, kp_set, kp_tgt, kp_offsets):

        if self.inference:
            make_func = self._make_source_target_inference

        else:
            make_func = self._make_source_target_teacher_forcing

        return make_func(prompt, kp_set, kp_tgt, kp_offsets)

    def _make_source_target_inference(self, prompt, kp_set, kp_tgt, kp_offsets):
        concat_src = torch.LongTensor(prompt + [self.sep] + kp_set)
        src_len = len(concat_src)

        kp_tgt_t = torch.LongTensor(kp_tgt)
        concat_tgt = torch.LongTensor(kp_tgt + [self.eos])

        kp_tgt_len = len(kp_tgt_t)
        ret_obj = {
            'concat_target': concat_tgt,
            'kp_target': kp_tgt_t,
            'kp_target_length': kp_tgt_len
        }

        kp_offset_target = torch.LongTensor(kp_offsets)
        ret_obj['model_input'] = concat_src
        ret_obj['src_length'] = src_len
        ret_obj['kp_offset_target'] = kp_offset_target
        return ret_obj

    def _make_source_target_teacher_forcing(self, prompt, kp_set, kp_tgt, kp_offsets):
        """Concatenate prompt, kp_set, kp_tgt"""
        concat_src = torch.LongTensor(prompt + [self.sep] + kp_set)
        src_len = len(concat_src)

        new_tgt = torch.LongTensor(kp_tgt)

        model_input = torch.cat((concat_src, torch.tensor([self.bok]), new_tgt))
        model_output = torch.cat((concat_src, new_tgt, torch.tensor([self.eos])))
        model_target = torch.cat((concat_src.new(concat_src.shape).fill_(self.pad),
                                  new_tgt,
                                  torch.tensor([self.eos])))
        seq_length = len(model_input)
        kp_offset_target = torch.LongTensor(kp_offsets)

        ret_obj = {
            'model_input': model_input,
            'true_output': model_output,
            'model_target': model_target,
            'src_length': src_len,
            'kp_tgt_length': len(kp_tgt),
            'seq_length': seq_length,
            'ntokens': seq_length,
            'kp_offset_target': kp_offset_target,
        }

        return ret_obj

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch"""
        if len(samples) == 0:
            return {}

        def merge(key, is_list=False):
            if is_list:
                res = []
                for i in range(len(samples[0][key])):
                    res.append(data_utils.collate_tokens(
                        [s[key][i] for s in samples], self.dictionary.pad_index,
                        self.dictionary.eos_index,
                    ))
                return res
            else:
                return data_utils.collate_tokens(
                    [s[key] for s in samples], self.dictionary.pad_index,
                    self.dictionary.eos_index
                )

        if self.inference:
            max_len = max([s['src_length'] for s in samples])
            input_ids = merge('model_input')

            ret_obj = {
                'id': [s['id'] for s in samples],
            }

            token_type_ids = torch.zeros((len(samples), max_len)).long()
            net_input = {'input_ids': input_ids,
                         'token_type_ids': token_type_ids}
            ret_obj['net_input'] = net_input
            ret_obj['src_length'] = torch.LongTensor([s['src_length'] for s in samples])
            ret_obj['kp_target'] = merge('kp_target')
            ret_obj['kp_target_length'] = torch.LongTensor([s['kp_target_length'] for s in samples])
            ret_obj['offset_target'] = merge('kp_offset_target')
            return ret_obj

        max_len = max([s['seq_length'] for s in samples])
        token_type_ids = np.ones((len(samples), max_len))
        kp_tgt_mask = np.zeros((len(samples), max_len))

        for ix, s in enumerate(samples):
            cur_token_type_ids = [0] * s['src_length'] + [1] * (s['seq_length'] - s['src_length'])
            token_type_ids[ix][:len(cur_token_type_ids)] = cur_token_type_ids
            kp_tgt_mask[ix][s['src_length'] + 1: s['src_length'] + s['kp_tgt_length'] + 1] = 1

        token_type_ids = torch.LongTensor(token_type_ids)
        kp_tgt_mask = torch.LongTensor(kp_tgt_mask)
        net_input = {'input_ids': merge('model_input'),
                     'kp_tgt_mask': kp_tgt_mask,
                     'token_type_ids': token_type_ids}

        ret_obj = {
            'id': [s['id'] for s in samples],
            'ntokens': sum(s['ntokens'] for s in samples),
            'net_input': net_input,
            'model_output': merge('true_output'),
            'model_target': merge('model_target'),
            'src_length': torch.LongTensor([s['src_length'] for s in samples]),
            'seq_length': torch.LongTensor([s['seq_length'] for s in samples]),
            'offset_target': merge('kp_offset_target'),
        }
        return ret_obj

    def ordered_indices(self):
        if self.train and self.seed is None:
            return np.random.permutation(len(self))
        indices = np.arange(len(self))
        return indices[np.argsort(self.lengths[indices], kind='mergesort')]

