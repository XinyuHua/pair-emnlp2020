## Script to calculate topic signature (log-likelihood test)

import json
from collections import defaultdict
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def compute_likelihood_ratio(c_1, c_12, c_2, p, p_1, p_2, N):
    """From `https://www.cs.cmu.edu/~hovy/papers/00linhovy.pdf`
    """
    def log_L(k, n, x):
        return k * np.log(x) + (n-k) * np.log(1 - x)

    return log_L(c_12, c_1, p) \
            + log_L(c_2 - c_12, N - c_1, p) \
            - log_L(c_12, c_1, p_1) \
            - log_L(c_2 - c_12, N - c_1, p_2)

class TopicSignatureConstruction:

    def __init__(self, data_path):
        self.data_path = data_path

        self.lemma_data = []

        # word frequency for each document
        self.doc2freq = dict()

        # total number of words in each document
        self.doc_total_words = dict()

        # frequency for each unique token
        self.total_freq = defaultdict(int)

        # total number of occurred tokens
        self.total_words = 0

        # supply any stop word lists you want
        self.stopwords = [ln.strip().lower() for ln in open('./stopwords.txt')]


    def load_data(self):
        """
        Load text data and run tokenization.
        Assume each line in `self.data_path` is a json object, which contains:
            `id`: unique document id
            `text`: untokenized original document
        """
        for ln in open(self.data_path):
            cur_obj = json.loads(ln)
            cur_id = cur_obj['id']
            cur_text = cur_obj['text']
            cur_words = word_tokenize(cur_text)

            lowercased_words = []

            # remove punctuations and stopwords
            for word in cur_words:
                word = word.lower()
                if not str.isalnum(word):
                    continue
                
                if word in self.stopwords:
                    continue

                lowercased_words.append(word)


            for word in lowercased_words:
                self.total_freq[word] += 1
                self.total_words += 1

                if cur_id not in self.doc2freq:
                    self.doc2freq[cur_id] = defaultdict(int)
                self.doc2freq[cur_id][word] += 1

                if cur_id not in self.doc_total_words:
                    self.doc_total_words[cur_id] = 0
                self.doc_total_words[cur_id] += 1

        print(f'{len(self.doc_total_words)} documents loaded')


    def calculate_llr(self):
        """Calculate log-likelihood ratio"""
        self.doc_word2ratio = {doc_id: defaultdict(float) \
                                       for doc_id in self.doc2freq}

        N = self.total_words
        for doc_id in tqdm(self.doc2freq):
            for word in self.doc2freq[doc_id]:
                if self.total_freq[word] < 10: continue
                c_2 = self.total_freq[word]
                p = c_2 / N

                c_12 = self.doc2freq[doc_id][word]
                if c_12 == 0:
                    continue

                c_1 = self.doc_total_words[doc_id]
                p_1 = c_12 / c_1
                p_2 = (c_2 - c_12) / (N - c_1)
                if c_2 == c_12:
                    cur_ratio = 0
                else:
                    cur_ratio = -2 * compute_likelihood_ratio(c_1, c_12, c_2, p, p_1, p_2, N=N)
                self.doc_word2ratio[doc_id][word] = cur_ratio

    def write_to_disk(self):
        fout = open('loglikelihood_ratio.jsonl', 'w')

        for doc_id, w2ratio in self.doc_word2ratio.items():
            ret_obj = {'id': doc_id, 'ratio_ranked_words': []}
            for item in sorted(w2ratio.items(), key=lambda x: x[1], reverse=True):
                output_tuple = (item[0], item[1], self.doc2freq[doc_id][item[0]])
                ret_obj['ratio_ranked_words'].append(output_tuple)

            fout.write(json.dumps(ret_obj) + '\n')
        fout.close()


if __name__=='__main__':

    DATA_PATH = 'demo.jsonl'

    ts_construction = TopicSignatureConstruction(data_path=DATA_PATH)
    ts_construction.load_data()
    ts_construction.calculate_llr()
    ts_construction.write_to_disk()

