import numpy as np
from torch.utils.data import Dataset
import utils

# (op, kp, tgt)
DOMAIN_TO_MAX_SIZES = {
    'arggen': [33, 100, 140],
    'opinion': [13, 150, 243],
    'news': [15, 250, 335]
}

class BaseDataset(Dataset):

    def __init__(self, args, set_type, tokenizer, is_inference):
        super().__init__()

        self.domain = args.domain
        self.setup = args.setup
        self.tokenizer = tokenizer
        self.set_type = set_type
        self.is_inference = is_inference

        self.max_prompt_len = args.max_prompt_len if args.max_prompt_len is not None else DOMAIN_TO_MAX_SIZES[self.domain][0]
        self.max_kp_len = args.max_kp_len if args.max_kp_len is not None else DOMAIN_TO_MAX_SIZES[self.domain][1]
        self.max_tgt_len = args.max_tgt_len if args.max_tgt_len is not None else DOMAIN_TO_MAX_SIZES[self.domain][2]

        self.sep_tok = '<s>'
        self.sep_idx = 0

        self.bok_tok = '<s>'
        self.bok_idx = 0

        self.bos_tok = '<s>'
        self.bos_idx = 0

        self.pad_tok = '<pad>'
        self.pad_idx = 1

        self.mask_tok = '<mask>'
        self.mask_idx = 50264

        self.eos_tok = '</s>'
        self.eos_idx = 2

        self.ID = []
        self.source = []
        self.target = []
        if self.is_inference:
            self.kp_set = []
            self.use_system_plan = args.use_system_plan
        self.random = np.random.RandomState(42)

    def __len__(self):
        return len(self.ID)

    def load_raw_dataset(self, path):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

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
            src_len=[s['src_len'] for s in samples]
        )

        if not self.is_inference or not self.use_system_plan:
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

        if self.is_inference:
            ret_obj['kp_set'] = [s['kp_set'] for s in samples]

        if 'template' in samples[0]:
            ret_obj['template'] = merge('template')

        return ret_obj
