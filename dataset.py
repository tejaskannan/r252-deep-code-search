
import numpy as np

from dpu_utils.mlutils import Vocabulary
from constants import *
from utils import flatten, load_data_file

METHOD_NAMES = "method_names"
METHOD_APIS = "method_apis"
METHOD_TOKENS = "method_tokens"
JAVADOC = "javadoc"

METHOD_NAME_LENGTHS = "method_name_lengths"
METHOD_API_LENGTHS = "method_api_lengths"
METHOD_TOKEN_LENGTHS = "method_token_lengths"
JAVADOC_LENGTHS = "javadoc_lengths"


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

        self.max_seq_length = max_seq_length

        all_data = [self.train_data[METHOD_NAMES], self.train_data[METHOD_APIS],
                    self.train_data[METHOD_TOKENS], self.train_data[JAVADOC]]
        all_tokens = set(flatten(all_data))

        self.vocabulary = Vocabulary.create_vocabulary(all_tokens,
                                                       max_vocab_size,
                                                       count_threshold=1,
                                                       add_pad=True)

        self.train_tensors = self._tensorize_data(self.train_data)
        self.valid_tensors = self._tensorize_data(self.valid_data)

    # train = False means we are using validation data
    def make_mini_batches(self, batch_size, train=True):
        combined = []

        tensor_dict = self.train_tensors if train else self.valid_tensors

        combined = list(zip(tensor_dict[METHOD_NAMES], tensor_dict[METHOD_APIS],
                            tensor_dict[METHOD_TOKENS], tensor_dict[JAVADOC],
                            tensor_dict[METHOD_NAME_LENGTHS], tensor_dict[METHOD_API_LENGTHS],
                            tensor_dict[METHOD_TOKEN_LENGTHS], tensor_dict[JAVADOC_LENGTHS]))

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

        return Batch(name_batches, api_batches, token_batches, javadoc_batches,
                     name_length_batches, api_length_batches, token_length_batches,
                     javadoc_length_batches)

    def _tensorize_data(self, data_dict):

        name_tensors = []
        api_tensors = []
        token_tensors = []
        javadoc_tensors = []

        name_lengths = []
        api_lengths = []
        token_lengths = []
        javadoc_lengths = []

        method_names = data_dict[METHOD_NAMES]
        method_api_calls = data_dict[METHOD_APIS]
        method_tokens = data_dict[METHOD_TOKENS]
        javadoc = data_dict[JAVADOC]

        for i in range(0, len(method_names)):
            name_vec = self.vocabulary.get_id_or_unk_multiple(method_names[i],
                                                              pad_to_size=self.max_seq_length)

            name_lengths.append(min(len(method_names[i]), self.max_seq_length))
            name_tensors.append(name_vec)

            api_vec = self.vocabulary.get_id_or_unk_multiple(method_api_calls[i],
                                                             pad_to_size=self.max_seq_length)
            api_lengths.append(min(len(method_api_calls[i]), self.max_seq_length))
            api_tensors.append(api_vec)

            token_vec = self.vocabulary.get_id_or_unk_multiple(method_tokens[i],
                                                               pad_to_size=self.max_seq_length)
            token_lengths.append(min(len(method_tokens[i]), self.max_seq_length))
            token_tensors.append(token_vec)

            javadoc_vec = self.vocabulary.get_id_or_unk_multiple(javadoc[i],
                                                                 pad_to_size=self.max_seq_length)
            javadoc_lengths.append(min(len(javadoc[i]), self.max_seq_length))
            javadoc_tensors.append(javadoc_vec)

        return {
            METHOD_NAMES: np.array(name_tensors),
            METHOD_NAME_LENGTHS: np.array(name_lengths),
            METHOD_APIS: np.array(api_tensors),
            METHOD_API_LENGTHS: np.array(api_lengths),
            METHOD_TOKENS: np.array(token_tensors),
            METHOD_TOKEN_LENGTHS: np.array(token_lengths),
            JAVADOC: np.array(javadoc_tensors),
            JAVADOC_LENGTHS: np.array(javadoc_lengths)
        }


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
