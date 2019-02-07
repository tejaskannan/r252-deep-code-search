
import numpy as np

from dpu_utils.mlutils import Vocabulary
from parser import JAVADOC_FILE_NAME, METHOD_NAME_FILE_NAME
from parser import METHOD_API_FILE_NAME, METHOD_TOKENS_FILE_NAME

class Dataset:

    def __init__(self, data_dir, seq_length, max_vocab_size):
        self.method_names = self._load_data_file(data_dir + "/" + METHOD_NAME_FILE_NAME)
        self.method_api_calls = self._load_data_file(data_dir + "/" + METHOD_API_FILE_NAME)
        self.method_tokens = self._load_data_file(data_dir + "/" + METHOD_TOKENS_FILE_NAME)
        self.javadoc = self._load_data_file(data_dir + "/" + JAVADOC_FILE_NAME)

        assert len(self.method_names) == len(self.method_api_calls)
        assert len(self.method_tokens) == len(self.javadoc)
        assert len(self.method_names) == len(self.javadoc)

        self.seq_length = seq_length
        self.data_count = len(self.method_names)

        all_tokens = self._flatten([self.method_names, self.method_tokens,\
                                    self.method_api_calls, self.javadoc])
        self.vocabulary = Vocabulary.create_vocabulary(all_tokens,
                                                       max_vocab_size,
                                                       add_pad=True)

        self._tensorize_data(self.method_names, self.method_api_calls,
                             self.method_tokens, self.javadoc)




    def make_mini_batches(self, batch_size):
        combined = list(zip(self.name_tensors, self.api_tensors, \
                            self.token_tensors, self.javadoc_tensors, \
                            self.name_lengths, self.api_lengths, \
                            self.token_lengths, self.javadoc_lengths))
        np.random.shuffle(combined)

        names, apis, tokens, javadocs, name_lengths, \
               api_lengths, token_lengths, javadoc_lengths = zip(*combined)

        name_batches = []
        api_batches = []
        token_batches = []
        javadoc_batches = []

        name_length_batches = []
        api_length_batches = []
        token_length_batches = []
        javadoc_length_batches = []

        for index in range(0, self.data_count, batch_size):
            limit = index + batch_size
            name_batches.append(np.array(names[index:limit]))
            api_batches.append(np.array(apis[index:limit]))
            token_batches.append(np.array(tokens[index:limit]))
            javadoc_batches.append(np.array(javadocs[index:limit]))

            name_length_batches.append(name_lengths[index:limit])
            api_length_batches.append(api_lengths[index:limit])
            token_length_batches.append(token_lengths[index:limit])
            javadoc_length_batches.append(javadoc_lengths[index:limit])

        return Batch(name_batches, api_batches, token_batches, javadoc_batches, \
                     name_length_batches, api_length_batches, token_length_batches, \
                     javadoc_length_batches)

    def _tensorize_data(self, method_names, method_api_calls, method_tokens, javadoc):

        def pad(text):
            if len(text) > self.seq_length:
                return text[:self.seq_length]
            return np.pad(text,
                          (0, self.seq_length - len(text)),
                          'constant',
                          constant_values=0)

        self.name_tensors = []
        self.api_tensors = []
        self.token_tensors = []
        self.javadoc_tensors = []

        self.name_lengths = []
        self.api_lengths = []
        self.token_lengths = []
        self.javadoc_lengths = []

        for i in range(0, self.data_count):
            name_vec = self.vocabulary.get_id_or_unk_multiple(method_names[i])
            self.name_lengths.append(min(len(name_vec), self.seq_length))
            self.name_tensors.append(pad(name_vec))

            api_vec = self.vocabulary.get_id_or_unk_multiple(method_api_calls[i])
            self.api_lengths.append(min(len(api_vec), self.seq_length))
            self.api_tensors.append(pad(api_vec))

            token_vec = self.vocabulary.get_id_or_unk_multiple(method_tokens[i])
            self.token_lengths.append(min(len(token_vec), self.seq_length))
            self.token_tensors.append(pad(token_vec))

            javadoc_vec = self.vocabulary.get_id_or_unk_multiple(javadoc[i])
            self.javadoc_lengths.append(min(len(javadoc_vec), self.seq_length))
            self.javadoc_tensors.append(pad(javadoc_vec))


        self.name_tensors = np.array(self.name_tensors)
        self.api_tensors = np.array(self.api_tensors)
        self.token_tensors = np.array(self.token_tensors)
        self.javadoc_tensors = np.array(self.javadoc_tensors)

    def _load_data_file(_, file_name):
        dataset = []
        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()
                dataset.append(line.split())
        return dataset

    def _flatten(_, lists):
        flattened = []
        for token_list in lists:
            for lst in token_list:
                flattened += lst
        return flattened


class Batch:

    def __init__(self, name_batches, api_batches, token_batches, javadoc_batches,
                 name_len_batches, api_len_batches, token_len_batches, javadoc_len_batches):
        self.name_batches = name_batches
        self.api_batches = api_batches
        self.token_batches = token_batches
        self.javadoc_batches = javadoc_batches
        self.name_len_batches = name_len_batches
        self.api_len_batches = api_len_batches
        self.token_len_batches = token_len_batches
        self.javadoc_len_batches = javadoc_len_batches

        self.num_batches = len(name_batches)
