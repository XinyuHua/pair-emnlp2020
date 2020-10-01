from collections import Counter
import torch
import torch.nn.functional as F


def duplicate_encoder_out(encoder_out, bsz, beam_size):
    encoder_out['encoder_out'] = encoder_out['encoder_out'].unsqueeze(2).repeat(1, 1, beam_size, 1).view(-1, bsz * beam_size, encoder_out['encoder_out'].size(-1))
    if encoder_out['encoder_padding_mask'] is not None:
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size, -1)


def generate_step_with_prob(out):
    probs = F.softmax(out[0], dim=-1)
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs


def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y


def assign_multi_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y.view(-1)[i.view(-1).nonzero()]


def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y


def assign_multi_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]


def convert_tokens(dictionary, tokens):
    return ' '.join([dictionary[token] for token in tokens])


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def extract_src_and_kp_from_input_ids(input_ids, task_dict):
    """Base on [SEP] and [unused1], extract and segment src-kp and src-input"""
    results = []
    for batch, ids in enumerate(input_ids):
        ids = ids.tolist()
        toks = task_dict.convert_ids_to_tokens(ids)
        src_end = False
        cur_result = {'src': [], 'kp': []}
        for w in toks:
            if w == '[SEP]' and not src_end:
                src_end = True
                cur_kp = []
                continue

            if w == '[unused1]':
                cur_result['kp'].append(cur_kp)
                break

            if not src_end:
                cur_result['src'].append(w)
            elif w == '[SEP]':
                if len(cur_kp) > 0:
                    cur_result['kp'].append(cur_kp)
                cur_kp = []
            else:
                cur_kp.append(w)
        results.append(cur_result)
    return results


def get_allowed_kp_wids(input_ids, eog):
    """For each input, get the vocabulary for KP that will later be used
    to enforce copy for kp-gen.

    Args:
        input_ids (torch.Tensor()): of size [batch, max_len]
        eog_id: (int)
    Returns:
        allowed_kp_wids (list[set]): a list of word id set, only these
            words are allowed during generating kp
    """
    allowed_kp_wids = [set() for _ in input_ids]
    for batch, item in enumerate(input_ids):
        cur_seq = item.tolist()
        kp_start = cur_seq.index(102) + 1
        for w in cur_seq[kp_start:]:
            if w > 0:
                allowed_kp_wids[batch].add(w)
        allowed_kp_wids[batch].add(eog)
    return allowed_kp_wids


def enforce_copy(disabled_ids, allowed_wids, kp_pred_unfinished, vocab_size):
    """Update disabled_ids to satisfy constraints of KP vocabulary.

    Args:
        disabled_ids (List[List]): the word ids that's disabled for each
            sample.
        allowed_wids (List[set]): the word ids that's from the input kp's
            vocabulary, allowed to be generated
        kp_pred_unfinished (torch.Tensor): of shape [batch, 1], indicating
            whether each sample has finished generating keyphrase (0 for
            finished, 1 for unfinished).
        vocab_size (int): maximum vocabulary size
    Returns:
        disabled_ids (List[List]): updated version of disabled_ids
    """

    for batch in range(len(allowed_wids)):
        if kp_pred_unfinished[batch]:
            indices_to_remove = [i for i in range(vocab_size) \
                                 if i not in allowed_wids[batch]]
            disabled_ids[batch].update(indices_to_remove)
    return disabled_ids


def update_kp_used_freq(kp_used_freq, input_ids, kp_pred_unfinished):
    """At the end of each step, we reset and re-calculate keyprhase usage"""
    for batch, kp_counter in enumerate(kp_used_freq):
        # reset all freq to 0
        for kp in kp_used_freq[batch]:
            kp_used_freq[batch][kp] = 0

        if kp_pred_unfinished[batch]:
            gen_so_far = input_ids[batch].tolist()
            gen_kp = gen_so_far[gen_so_far.index(1) + 1:]

            # for each keyphrase, count their occurrence
            for kp in kp_used_freq[batch]:
                if len(kp) == 0: continue
                for wid, w in enumerate(gen_kp):
                    if w != kp[0]: continue
                    if wid + len(kp) > len(gen_kp): continue
                    if gen_kp[wid:wid + len(kp)] == list(kp):
                        kp_used_freq[batch][kp] += 1
    return kp_used_freq


def enforce_non_ngram_repeat(disabled_ids, input_ids, disable_ngram_repeat):
    for ix, seq in enumerate(input_ids):
        seq = seq.tolist()
        if 105 not in seq:
            continue
        tgt = seq[seq.index(105):]
        if len(tgt) <= disable_ngram_repeat:
            continue

        gen_ngrams = set()
        for i in range(0, len(tgt) - disable_ngram_repeat):
            cur_ngram = tgt[i:i + disable_ngram_repeat]
            if 102 in cur_ngram: continue # do not consider [SEP]
            gen_ngrams.add(tuple(cur_ngram))

        cur_disabled = set()
        prev_k_tok = tuple(tgt[1 - disable_ngram_repeat:])
        for ngram in gen_ngrams:
            if ngram[:-1] == prev_k_tok:
                cur_disabled.add(ngram[-1])
        for item in cur_disabled:
            disabled_ids[ix].update([item])
    return disabled_ids


def setup_kp_stats(input_ids):
    """When kp gen has freq limit, produce kp_used_freq and kp_unique_word
    to assist imposing the constraints.

    Args:
        input_ids (torch.Tensor): of shape [batch, max_src_len]
    Returns:
        kp_used_freq
        kp_unique_word
    """
    kp_used_freq = [Counter() for _ in range(input_ids.shape[0])]
    # store words that are unique to each keyphrase
    kp_unique_words = [dict() for _ in range(input_ids.shape[0])]

    for batch, item in enumerate(input_ids):
        cur_seq = item.tolist()
        kp_start = cur_seq.index(102) + 1
        cur_kp_words = cur_seq[kp_start:]
        word2kp_freq = Counter()
        while 102 in cur_kp_words:
            cur_ph_end = cur_kp_words.index(102)
            cur_ph = cur_kp_words[:cur_ph_end]
            for w in cur_ph:
                word2kp_freq[w] += 1

            kp_used_freq[batch][tuple(cur_ph)] = 0
            cur_kp_words = cur_kp_words[cur_ph_end + 1:]

        for kp in kp_used_freq[batch]:
            cur_unique = [w for w in kp if word2kp_freq[w] == 1]
            if kp not in kp_unique_words[batch]:
                kp_unique_words[batch][kp] = []
            kp_unique_words[batch][kp].append(cur_unique)
    return kp_used_freq, kp_unique_words


def enforce_kp_times(disabled_ids, kp_used_freq, kp_unique_words,
                      kp_pred_unfinished, kp_gen_max_time):
    """Disabled tokens that result in keyphrases more than their limits.

    Args:
        kp_used_freq (List[dict]): each element maps the keyphrase (tuple)
            into how many times they have been generated so far.
        kp_unique_words
    """

    for batch, kp_counter in enumerate(kp_used_freq):
        forbidden_words = []
        for kp, freq in kp_counter.items():
            if freq >= kp_gen_max_time:
                cur_kp_unique_words = kp_unique_words[batch][kp][0]
                forbidden_words.extend(cur_kp_unique_words)
        if kp_pred_unfinished[batch]:
            disabled_ids[batch].update(forbidden_words)
    return disabled_ids


def calculate_banned_tokens(input_ids, disable_ngram_repeat):
    """calculate which words are not allowed for next token.

    Args:
        input_ids (batch_size, length): only starts disabling after [unused101] (BOS)
        disable_ngram_repeat (int)
    Return:
        disabled_ids (batch_size, None)
    """
    disabled_ids = [[] for _ in range(input_ids.shape[0])]
    for ix, seq in enumerate(input_ids):
        seq = seq.tolist()
        if 105 not in seq:
            continue
        tgt = seq[seq.index(105):]
        if len(tgt) <= disable_ngram_repeat:
            continue
        gen_ngrams = set()
        for i in range(0, len(tgt) - disable_ngram_repeat):
            cur_ngram = tgt[i:i + disable_ngram_repeat]
            gen_ngrams.add(tuple(cur_ngram))

        cur_disabled = set()
        prev_k_tok = tuple(tgt[1 - disable_ngram_repeat:])
        for ngram in gen_ngrams:
            if ngram[:-1] == prev_k_tok:
                cur_disabled.add(ngram[-1])
        for item in cur_disabled:
            disabled_ids[ix].append(item)
    return disabled_ids
