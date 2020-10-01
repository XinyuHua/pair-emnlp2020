from transformers import BertTokenizer
import torch

class BertDictionary(object):
    """Wrapper for BertTokenizer"""
    def __init__(self, bert_model_name='bert-base-cased',
                 eos_token='[unused100]', bos_token='[unused101]', bok_token='[unused1]'):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.tokenizer.eos_token = eos_token
        self.tokenizer.bos_token = bos_token
        self.tokenizer.bok_token = bok_token
        self.tokenizer.bok_token_id = self.tokenizer.convert_tokens_to_ids([bok_token])[0]

        self.unk_word = self.tokenizer._unk_token
        self.pad_word = self.tokenizer._pad_token
        self.mask_word = self.tokenizer._mask_token
        self.sep_word = self.tokenizer._sep_token
        self.bos_word = self.tokenizer._bos_token
        self.eos_word = self.tokenizer._eos_token
        self.bok_word = bok_token

        self.unk_index = self.tokenizer.unk_token_id
        self.mask_index = self.tokenizer.mask_token_id
        self.pad_index = self.tokenizer.pad_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.sep_index = self.tokenizer.sep_token_id
        self.bok_index = self.tokenizer.bok_token_id

        self.indices = self.tokenizer.vocab
        self.symbols = self.tokenizer.ids_to_tokens

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        return len(self.symbols)

    def index(self, sym):
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_tokens_to_string(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def encode(self, string, add_special_tokens):
        return self.tokenizer.encode(string,
                                     add_special_tokens=add_special_tokens)

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def bok(self):
        return self.bok_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def sep(self):
        return self.sep_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def mask(self):
        """Helper to get index of mask symbol"""
        return self.mask_index

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

