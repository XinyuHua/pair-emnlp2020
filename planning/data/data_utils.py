import os
import numpy as np
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib


def find_phrase_boundaries(kp_src, kp_tgt, kp_offsets):
    """Given a sequence of keyphrase words, find their first words"""
    bio_seq = []
    src_kw_first2seq = dict()
    cur_kw = []
    for item in kp_src:
        if item == 102:
            if len(cur_kw) == 0: continue
            first_word = cur_kw[0]
            if not first_word in src_kw_first2seq:
                src_kw_first2seq[first_word] = []
            src_kw_first2seq[first_word].append(tuple(cur_kw))
            cur_kw = []
        else:
            cur_kw.append(item)

    def _update_by_offset(kp_tgt_ptr):
        cand_span = [kp_offsets[kp_tgt_ptr]]
        for step in range(kp_tgt_ptr + 1, len(kp_offsets)):
            if kp_offsets[step] == cand_span[-1] + 1:  # consecutive
                cand_span.append(kp_offsets[step])
            else:
                break
        return cand_span

    kp_tgt_ptr = 0
    while kp_tgt_ptr < len(kp_tgt):
        cur_w = kp_tgt[kp_tgt_ptr]
        if cur_w == 102:  # reaches SEP
            bio_seq.append('O')
            kp_tgt_ptr += 1

        elif cur_w in src_kw_first2seq:
            all_possible_seqs = src_kw_first2seq[cur_w]
            longest_matched = []
            for seq in all_possible_seqs:
                if len(seq) + kp_tgt_ptr > len(kp_tgt):
                    seq = seq[:len(kp_tgt) - kp_tgt_ptr]
                # cur_offsets_ = kp_offsets[kp_tgt_ptr:kp_tgt_ptr + len(seq)]
                # if cur_offsets_ == list(range(cur_offsets_[0], cur_offsets_[0] + len(seq))) \
                #         and seq == tuple(kp_tgt[kp_tgt_ptr:kp_tgt_ptr + len(seq)]):
                if seq == tuple(kp_tgt[kp_tgt_ptr: kp_tgt_ptr + len(seq)]):
                    if len(seq) > len(longest_matched):
                        longest_matched = seq

            if len(longest_matched) == 0:
                cand_span = _update_by_offset(kp_tgt_ptr)
                bio_seq.append('B')
                bio_seq.extend(['I'] * (len(cand_span) - 1))
                kp_tgt_ptr += len(cand_span)
            else:
                bio_seq.append('B')
                bio_seq.extend(['I'] * (len(longest_matched) - 1))
                kp_tgt_ptr += len(longest_matched)

        else:
            # possibly because the kp-src was cut-off, resort to kp-offsets
            cand_span = _update_by_offset(kp_tgt_ptr)
            bio_seq.append('B')
            bio_seq.extend(['I'] * (len(cand_span) - 1))
            kp_tgt_ptr += len(cand_span)

    assert len(kp_tgt) == len(bio_seq)
    bio_mask = [0 if bio == 'I' else org_offset + 1 for bio, org_offset in zip(bio_seq, kp_offsets)]
    return bio_mask


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)

def filter_by_size(indices, size_fn, max_positions, raise_exception=False):
    """
    Filter indices based on their size.

    Args:
        indices (List[int]): ordered list of dataset indices
        size_fn (callable): function that returns the size of a given index
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).
    """
    def check_size(idx):
        cur_size = size_fn(idx)
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, tuple):
            leq = [d_size <= m_size for d_size, m_size in zip(cur_size, max_positions)]
            return all(leq)


    ignored = []
    itr = collect_filtered(check_size, indices, ignored)

    for idx in itr:
        if len(ignored) > 0 and raise_exception:
            raise Exception((
                'Size of sample #{} is invalid (={}) since max_positions={}, '
                'skip this example with --skip-invalid-size-inputs-valid-test'
            ).format(ignored[0], size_fn(ignored[0]), max_positions))
        yield idx

    if len(ignored) > 0:
        print((
            '| WARNING: {} samples have invalid sizes and will be skipped, '
            'max_positions={}, first few sample ids={}'
        ).format(len(ignored), max_positions, ignored[:10]))

def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    max_tokens = max_tokens if max_tokens is not None else float('Inf')
    max_sentences = max_sentences if max_sentences is not None else float('Inf')
    bsz_mult = required_batch_size_multiple

    batch = []

    def is_batch_full(num_tokens):
        if len(batch) == 0:
            return False
        if len(batch) == max_sentences:
            return True
        if num_tokens > max_tokens:
            return True
        return False

    sample_len = 0
    sample_lens = []
    for idx in indices:
        sample_lens.append(num_tokens_fn(idx))
        sample_len = max(sample_len, sample_lens[-1])
        assert sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len
        if is_batch_full(num_tokens):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            yield batch[:mod_len]
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0

        batch.append(idx)

    if len(batch) > 0:
        yield batch

@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)