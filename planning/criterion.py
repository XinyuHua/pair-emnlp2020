import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import utils
import numpy as np

class CrossEntropyCriterionWithOffset(_Loss):

    def __init__(self, args, task):
        super().__init__()
        self.args = args
        self.task = task
        self.padding_idx = task.dictionary.pad()

    @classmethod
    def build_criterion(cls, args, task):
        return cls(args, task)

    def forward(self, model, sample):
        net_output = model(**sample['net_input'])
        total_loss, nll_loss, ko_loss, corr, predicted, to_print, nkp_words, \
        corr_offset, total_offset, offset_print = self.compute_loss(model, net_output, sample)
        sample_size = sample['model_output'].size(0)
        logging_output = {
            'total_loss': utils.item(total_loss.data),
            'nll_loss': utils.item(nll_loss.data),
            'kp_offset_loss': utils.item(ko_loss.data),
            'ntokens': predicted,
            'nkp_tokens': nkp_words,
            'sample_size': sample_size,
        }
        if total_offset > 0:
            logging_output['offset_accuracy'] = corr_offset / total_offset
        return (total_loss, nll_loss, ko_loss, sample_size, logging_output,
               to_print, offset_print)

    def smooth_offset_prediction(self, lprobs, target, ignore_index=0):
        """
        lprobs: [batch, C]
        target: [batch]
        """
        WS = self.args.smoothed_offset_window
        R = self.args.smoothed_offset_ratio
        C = lprobs.shape[-1]
        N = len(target)

        kernel = np.zeros((N, C))
        for ix in range(len(target)):
            central = target[ix].item()
            if ignore_index == central:
                continue
            lower = max(0, central - WS)
            upper = min(C - 1, central + WS)

            values = R ** np.arange(WS + 1)
            kernel[ix, central: upper + 1] = values[:upper - central + 1]
            kernel[ix, lower: central] = values[:central - lower + 1][:0:-1]
            kernel[ix] /= kernel[ix].sum()
        filtered_probs = torch.FloatTensor(kernel).cuda() * lprobs
        smoothed_loss = filtered_probs.sum(dim=-1)
        return -1 * smoothed_loss.sum()


    def compute_loss(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output['lm_out'], log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        pwo_lprobs = model.get_normalized_probs(
            net_output['offset_out'],
            log_probs=True)

        pwo_lprobs = pwo_lprobs.view(-1, pwo_lprobs.size(-1))
        pwo_target = sample['offset_target'].view(-1)
        if self.args.smoothed_offset_window == 0:
            wo_loss = F.nll_loss(pwo_lprobs,
                                 pwo_target,
                                 ignore_index=0,
                                 reduction='sum')
        else:
            wo_loss = self.smooth_offset_prediction(pwo_lprobs, pwo_target)

        nkp_words = sample['offset_target'].ne(0).sum().item()

        target = sample['model_target']
        target = target.view(-1)
        nll_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum',
        )

        total_loss = nll_loss + wo_loss * self.args.keyphrase_offset_loss_alpha


        # DEBUG INFO
        real_probs = model.get_normalized_probs(net_output['lm_out'], log_probs=False)
        pred_probs, pred_tok_ids_ = torch.max(real_probs, dim=-1)
        pred_probs = pred_probs.detach().cpu().numpy()
        pred_tok_ids = pred_tok_ids_.detach().cpu().numpy()
        target_2d = sample['model_target']

        # calculate accuracy
        # we consider accuracy as the number of predicted tokens that match
        # the gold-standard
        truth_probs = real_probs.gather(dim=-1, index=target_2d.unsqueeze(-1))
        truth_probs_ = truth_probs.detach().cpu().numpy()
        target_ = target_2d.detach().cpu().numpy()

        total_predicted_toks = 0
        total_correct_toks = 0
        dec_input = sample['net_input']['input_ids'].detach().cpu().numpy()
        for ix, cur_target in enumerate(target_):
            cur_src_len = sample['src_length'][ix].item()
            cur_seq_len = sample['seq_length'][ix].item()
            cur_target = cur_target[cur_src_len:cur_seq_len]
            cur_pred = pred_tok_ids[ix][cur_src_len:cur_seq_len]
            cur_prob = truth_probs_[ix][cur_src_len:cur_seq_len]
            total_predicted_toks += len(cur_target)
            total_correct_toks += sum(cur_pred == cur_target)
            to_print = ''

            if not self.args.quiet:
                cur_pred_prob = pred_probs[ix][cur_src_len:cur_seq_len]
                cur_dec_in = dec_input[ix][cur_src_len:cur_seq_len]
                cur_dec_in_toks = [self.task.task_dict[t] for t in cur_dec_in]
                cur_pred_toks = [self.task.task_dict[t] for t in cur_pred]
                cur_truth_toks = [self.task.task_dict[t] for t in cur_target]
                for inp, pred, truth, p, pp in zip(cur_dec_in_toks, cur_pred_toks, cur_truth_toks, cur_prob,
                                                   cur_pred_prob):
                    if truth == '[PAD]':
                        to_print += 'IN:{} '.format(inp)
                    else:
                        if pred == truth:
                            to_print += '\033[32m {}({:.2f}) \033[m '.format(pred, 100 * pp)
                        else:
                            color = '\033[31m'
                            to_print += '{}(P:{}({:.2f}) T:{}({:.2f})) \033[m  '.format(color, pred, 100 * pp, truth,
                                                                                        100 * p[0])
                print(to_print + '\n')

        total_predicted_offsets = 0
        total_corr_offsets = 0
        kp_offset_to_print = ''

        real_ko_probs = model.get_normalized_probs(net_output['offset_out'],
                                                   log_probs=False)
        pred_ko_probs, pred_offset_ids = torch.max(real_ko_probs, dim=-1)
        target_2d = sample['offset_target']
        target_ = target_2d.detach().cpu().numpy()

        for ix, cur_target in enumerate(target_):
            cur_pred = pred_offset_ids[ix].detach().cpu().numpy()
            for wix in range(len(cur_pred)):
                if cur_target[wix] == 0: continue
                total_predicted_offsets += 1
                pred = cur_pred[wix]
                truth = cur_target[wix]

                if truth == pred:
                    kp_offset_to_print += '\033[32m {} \033[m '.format(pred)
                    total_corr_offsets += 1
                else:
                    color = '\033[31m'
                    kp_offset_to_print += '{}(P:{} T:{}) \033[m  '.format(color, pred, truth)
            kp_offset_to_print += '\n'

        return total_loss, nll_loss, wo_loss, total_correct_toks, \
               total_predicted_toks, to_print, nkp_words, total_corr_offsets, \
               total_predicted_offsets, kp_offset_to_print

    @staticmethod
    def grad_denom(sample_sizes):
        return sum(sample_sizes)