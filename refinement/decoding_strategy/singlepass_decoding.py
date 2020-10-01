import torch
import torch.nn.functional as F
from .strategy_utils import top_k_top_p_filtering
import utils

from decoding_strategy import BaseDecoding

class SinglePassDecoding(BaseDecoding):

    def generate(self, model, batch):

        net_input = utils.move_to_cuda(batch['net_input'])
        encoder_input_ids = net_input['input_ids']
        encoder_attn_mask = net_input['attention_mask']
        batch_size = encoder_input_ids.shape[0]

        encoder = model.get_encoder()
        encoder_outputs = encoder(encoder_input_ids,
                                  attention_mask=encoder_attn_mask)

        # create empty decoder_input_ids
        input_ids = torch.full(
            (batch_size, 1),
            self.decoder_bos_idx,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        cur_len = 1
        probs = [[] for _ in range(batch_size)]

        unfinished_sents = input_ids.new(batch_size).fill_(1)

        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        while cur_len < self.domain_to_max_len[self.domain]:
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              past=past,
                                                              attention_mask=encoder_attn_mask)

            outputs = model(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            past = outputs[1]


            if self.do_sampling:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if self.temperature != 1.0:
                    next_token_logits = next_token_logits / self.temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
                # Sample
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(next_token_logits, dim=-1)


            chosen_token_probs = next_token_probs.gather(1, next_token.view(-1, 1))
            for b in range(batch_size):
                probs[b].append(chosen_token_probs[b, 0].item())

            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (self.pad_idx) * (1 - unfinished_sents)

            if not self.quiet:
                output_str = ''
                for b in range(batch_size):
                    w = self.tokenizer.convert_ids_to_tokens([tokens_to_add[b]])[0]
                    p = probs[b][-1]
                    output_str += '{:>12}({:.2f})|'.format(w, 100 * p)
                if cur_len == 1:
                    print('=' * 50)
                print('step={:<3d}|{}'.format(cur_len, output_str))

            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            eos_in_sents = tokens_to_add == self.eos_idx
            unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break
            cur_len = cur_len + 1

        return input_ids, probs


