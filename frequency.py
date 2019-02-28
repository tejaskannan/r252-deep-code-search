import numpy as np
from constants import *
from utils import flatten, softmax

class TokenFrequency:

    def __init__(self, tokens_lists, vocabulary):
        self.documents = []
        for i in range(len(tokens_lists[0])):
            doc = []
            for token_list in tokens_lists:
                doc += [t for t in token_list[i] if not vocabulary.is_unk(t)]
            self.documents.append(doc)
        self.vocabulary = vocabulary

        # mapping of word to number of uses
        self.doc_counts = self._create_doc_counts()

        self.num_docs = float(len(self.documents))

    # method_lists is a list of lists
    def tf_idf_multiple(self, doc_lists, pad_to_size=-1):
        doc_tokens = []
        for lst in doc_lists:
            doc_tokens += lst

        scores = []
        for lst in doc_lists:
            score_arr = [self.tf_idf(word, doc_tokens) for word in lst]
            if pad_to_size != -1:
                if len(score_arr) < pad_to_size:
                    score_arr = np.pad(score_arr, (0, pad_to_size - len(score_arr)), mode='constant',
                                       constant_values=-BIG_NUMBER)
                else:
                    score_arr = score_arr[0:pad_to_size]

            scores.append(softmax(score_arr))
        return scores

    def tf_idf(self, word, doc):
        word = word.strip().lower()
        if self.vocabulary.is_unk(word) or (not word in self.doc_counts):
            return -BIG_NUMBER

        word_count = np.sum([int(w == word) for w in doc])
        word_freq = float(word_count) / float(len(doc))
        doc_freq = self.doc_counts[word] / self.num_docs

        return word_freq * np.log(self.num_docs / self.doc_counts[word])

    def _create_doc_counts(self):
        counts = {}
        for doc in self.documents:
            for token in set(doc):
                if not token in counts:
                    counts[token] = 0
                counts[token] += 1
        return counts
