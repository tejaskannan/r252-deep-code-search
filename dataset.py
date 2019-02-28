
import numpy as np

from dpu_utils.mlutils import Vocabulary
from constants import *
from utils import flatten, load_data_file
from frequency import TokenFrequency

METHOD_NAMES = "method_names"
METHOD_APIS = "method_apis"
METHOD_TOKENS = "method_tokens"
JAVADOC = "javadoc"


class Dataset:

    def __init__(self, train_dir, valid_dir, max_seq_length, max_vocab_size):

        self.train_data = {
            METHOD_NAMES: load_data_file(train_dir + METHOD_NAME_FILE_NAME),
            METHOD_APIS: load_data_file(train_dir + METHOD_API_FILE_NAME),
            METHOD_TOKENS: load_data_file(train_dir + METHOD_TOKENS_FILE_NAME),
            JAVADOC: load_data_file(train_dir + JAVADOC_FILE_NAME)
        }

        self.valid_data = {
            METHOD_NAMES: load_data_file(valid_dir + METHOD_NAME_FILE_NAME),
            METHOD_APIS: load_data_file(valid_dir + METHOD_API_FILE_NAME),
            METHOD_TOKENS: load_data_file(valid_dir + METHOD_TOKENS_FILE_NAME),
            JAVADOC: load_data_file(valid_dir + JAVADOC_FILE_NAME)
        }

        methods_list = [self.train_data[METHOD_NAMES], self.train_data[METHOD_APIS],
                        self.train_data[METHOD_TOKENS]]
        javadoc_list = [self.train_data[JAVADOC]]

        all_tokens = flatten(methods_list + javadoc_list)

        self.max_seq_length = max_seq_length
        self.max_vocab_size = max_vocab_size

        self.vocabulary = Vocabulary.create_vocabulary(all_tokens,
                                                       max_vocab_size,
                                                       count_threshold=1,
                                                       add_pad=True)


        self.method_frequency = TokenFrequency(methods_list, self.vocabulary)
        self.javadoc_frequency = TokenFrequency(javadoc_list, self.vocabulary)

        self.train_tensors = self._tensorize_data(self.train_data)
        self.valid_tensors = self._tensorize_data(self.valid_data)

    # train = False means we are using validation data
    def make_mini_batches(self, batch_size, train=True):
        combined = []

        tensor_dict = self.train_tensors if train else self.valid_tensors

        combined = list(zip(tensor_dict[METHOD_NAMES], tensor_dict[METHOD_APIS],
                            tensor_dict[METHOD_TOKENS], tensor_dict[JAVADOC],
                            tensor_dict[METHOD_NAME_LENGTHS], tensor_dict[METHOD_API_LENGTHS],
                            tensor_dict[METHOD_TOKEN_LENGTHS], tensor_dict[JAVADOC_LENGTHS],
                            tensor_dict[METHOD_NAME_FREQ], tensor_dict[METHOD_API_FREQ],
                            tensor_dict[METHOD_TOKEN_FREQ], tensor_dict[JAVADOC_FREQ]))

        np.random.shuffle(combined)

        names, apis, tokens, javadocs, name_lengths, \
            api_lengths, token_lengths, javadoc_lengths, \
            name_freq, api_freq, token_freq, javadoc_freq = zip(*combined)

        name_batches = []
        api_batches = []
        token_batches = []
        javadoc_batches = []

        name_length_batches = []
        api_length_batches = []
        token_length_batches = []
        javadoc_length_batches = []

        name_freq_batches = []
        api_freq_batches = []
        token_freq_batches = []
        javadoc_freq_batches = []

        assert len(names) == len(apis)
        assert len(tokens) == len(javadocs)
        assert len(apis) == len(tokens)

        for index in range(0, len(names), batch_size):
            limit = index + batch_size
            name_batches.append(np.array(names[index:limit]))
            api_batches.append(np.array(apis[index:limit]))
            token_batches.append(np.array(tokens[index:limit]))
            javadoc_batches.append(np.array(javadocs[index:limit]))

            name_length_batches.append(name_lengths[index:limit])
            api_length_batches.append(api_lengths[index:limit])
            token_length_batches.append(token_lengths[index:limit])
            javadoc_length_batches.append(javadoc_lengths[index:limit])

            name_freq_batches.append(name_freq[index:limit])
            api_freq_batches.append(api_freq[index:limit])
            token_freq_batches.append(token_freq[index:limit])
            javadoc_freq_batches.append(javadoc_freq[index:limit])

        return Batch(name_batches=name_batches,
                     api_batches=api_batches,
                     token_batches=token_batches,
                     javadoc_batches=javadoc_batches,
                     name_len_batches=name_length_batches,
                     api_len_batches=api_length_batches,
                     token_len_batches=token_length_batches,
                     javadoc_len_batches=javadoc_length_batches,
                     name_freq_batches=name_freq_batches,
                     api_freq_batches=api_freq_batches,
                     token_freq_batches=token_freq_batches,
                     javadoc_freq_batches=javadoc_freq_batches)

    def create_tensor(self, sequence):
        # We ensure that we are using the tokens within the given sequence
        if type(sequence) is str:
            sequence = sequence.split(' ')

        seq_tensor = self.vocabulary.get_id_or_unk_multiple(sequence, pad_to_size=self.max_seq_length)
        seq_length = min(len(sequence), self.max_seq_length)
        return seq_tensor, seq_length

    def _tensorize_data(self, data_dict):

        name_tensors = []
        api_tensors = []
        token_tensors = []
        javadoc_tensors = []

        name_lengths = []
        api_lengths = []
        token_lengths = []
        javadoc_lengths = []

        name_freqs = []
        api_freqs = []
        token_freqs = []
        javadoc_freqs = []

        method_names = data_dict[METHOD_NAMES]
        method_api_calls = data_dict[METHOD_APIS]
        method_tokens = data_dict[METHOD_TOKENS]
        javadoc = data_dict[JAVADOC]

        for i in range(0, len(method_names)):
            name_vec, name_len = self.create_tensor(method_names[i])
            name_lengths.append(name_len)
            name_tensors.append(name_vec)

            api_vec, api_len = self.create_tensor(method_api_calls[i])
            api_lengths.append(api_len)
            api_tensors.append(api_vec)

            token_vec, token_len = self.create_tensor(method_tokens[i])
            token_lengths.append(token_len)
            token_tensors.append(token_vec)

            javadoc_vec, javadoc_len = self.create_tensor(javadoc[i])
            javadoc_lengths.append(javadoc_len)
            javadoc_tensors.append(javadoc_vec)

            method = [method_names[i], method_api_calls[i], method_tokens[i]]
            method_f = self.method_frequency.tf_idf_multiple(method)
            name_freqs.append(method_f[0])
            api_freqs.append(method_f[1])
            token_freqs.append(method_f[2])

            javadoc_f = self.javadoc_frequency.tf_idf_multiple([javadoc[i]])
            javadoc_freqs.append(javadoc_f[0])

        return {
            METHOD_NAMES: np.array(name_tensors),
            METHOD_NAME_LENGTHS: np.array(name_lengths),
            METHOD_NAME_FREQ: np.array(name_freqs),
            METHOD_APIS: np.array(api_tensors),
            METHOD_API_LENGTHS: np.array(api_lengths),
            METHOD_API_FREQ: np.array(api_freqs),
            METHOD_TOKENS: np.array(token_tensors),
            METHOD_TOKEN_LENGTHS: np.array(token_lengths),
            METHOD_TOKEN_FREQ: np.array(token_freqs),
            JAVADOC: np.array(javadoc_tensors),
            JAVADOC_LENGTHS: np.array(javadoc_lengths),
            JAVADOC_FREQ: np.array(javadoc_freqs)
        }


class Batch:

    def __init__(self, name_batches, api_batches, token_batches, javadoc_batches,
                 name_len_batches, api_len_batches, token_len_batches, javadoc_len_batches,
                 name_freq_batches, api_freq_batches, token_freq_batches, javadoc_freq_batches):
        self.name_batches = name_batches
        self.api_batches = api_batches
        self.token_batches = token_batches
        self.javadoc_batches = javadoc_batches
        self.name_len_batches = name_len_batches
        self.api_len_batches = api_len_batches
        self.token_len_batches = token_len_batches
        self.javadoc_len_batches = javadoc_len_batches
        self.name_freq_batches = name_freq_batches
        self.api_freq_batches = api_freq_batches
        self.token_freq_batches = token_freq_batches
        self.javadoc_freq_batches = javadoc_freq_batches
        self.num_batches = len(name_batches)
