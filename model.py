
import numpy as np
import tensorflow as tf

from dpu_utils.mlutils import Vocabulary

from parser import JAVADOC_FILE_NAME, METHOD_NAME_FILE_NAME
from parser import METHOD_API_FILE_NAME, METHOD_TOKENS_FILE_NAME

class Model:


    def __init__(self, train_dir="data"):
        # Intialize some hyperparameters
        self.train_frac = 0.8
        self.valid_frac = 0.2
        self.step_size = 0.01
        self.margin = 0.05
        self.max_vocab_size = 50000
        self.seq_length = 50
        self.rnn_units = 16
        self.batch_size = 128

        self.method_names = self._load_data_file(train_dir + "/" + METHOD_NAME_FILE_NAME)
        self.method_api_calls = self._load_data_file(train_dir + "/" + METHOD_API_FILE_NAME)
        self.method_tokens = self._load_data_file(train_dir + "/" + METHOD_TOKENS_FILE_NAME)
        self.javadoc = self._load_data_file(train_dir + "/" + JAVADOC_FILE_NAME)

        assert len(self.method_names) == len(self.method_api_calls)
        assert len(self.method_tokens) == len(self.javadoc)
        assert len(self.method_names) == len(self.javadoc)

        self.data_count = len(self.method_names)

        all_tokens = self._flatten([self.method_names, self.method_tokens,\
                                    self.method_api_calls, self.javadoc])
        self.vocabulary = Vocabulary.create_vocabulary(all_tokens, self.max_vocab_size, add_pad=True)

        # Vectorize strings using the vocabulary
        tensors = self._tensorize_data(self.method_names, self.method_api_calls,
                                       self.method_tokens, self.javadoc)
        self.name_tensors, self.api_tensors, self.token_tensors, self.javadoc_tensors = tensors

        self._sess = tf.Session(graph=tf.Graph())

        with self._sess.graph.as_default():
            self.name_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='name')
            self.api_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='api')
            self.token_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='token')
            self.javadoc_placehoder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='javadoc')

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.step_size)

            self._make_model()
    

    def train(self):

        with self._sess.graph.as_default():

            init_op = tf.variables_initializer(self._sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self._sess.run(init_op)

            for index in range(0, self.data_count, self.batch_size):
                end = index + self.batch_size
                name_batch = self.name_tensors[index:end]
                api_batch = self.api_tensors[index:end]
                token_batch = self.token_tensors[index:end]
                javadoc_batch = self.javadoc_tensors[index:end]

                feed_dict = {
                    self.name_placeholder: name_batch,
                    self.api_placeholder: api_batch,
                    self.token_placeholder: token_batch,
                    self.javadoc_placehoder: javadoc_batch
                }

                op_result = self._sess.run(self.loss_op, feed_dict=feed_dict)
                print(op_result)
                break



    def _make_model(self):

        # Method Name embedding
        one_hot_names = tf.one_hot(self.name_placeholder, depth=1, on_value=1.0, off_value=0.0, dtype=tf.float32)
        name_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_units)
        name_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_units)
        name_emb, name_state = tf.nn.bidirectional_dynamic_rnn(
                                        cell_fw=name_cell_fw,
                                        cell_bw=name_cell_bw,
                                        inputs=one_hot_names,
                                        dtype=tf.float32,
                                        scope="name-embedding")

        # Javadoc Embedding
        one_hot_javadoc = tf.one_hot(self.javadoc_placehoder, depth=1, on_value=1.0, off_value=0.0, dtype=tf.float32)
        jd_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_units)
        jd_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_units)
        jd_emb, jd_state = tf.nn.bidirectional_dynamic_rnn(
                                        cell_fw=jd_cell_fw,
                                        cell_bw=jd_cell_bw,
                                        inputs=one_hot_javadoc,
                                        dtype=tf.float32,
                                        scope="javadoc-embedding")

        print(name_emb)
        print(jd_emb)

        self.loss_op = tf.math.reduce_sum(
                tf.constant(self.margin, dtype=tf.float32) - \
                tf.losses.cosine_distance(labels=jd_emb, predictions=name_emb, axis=1)
            )



    def _tensorize_data(self, method_names, method_api_calls, method_tokens, javadoc):

        def pad(text):
            if len(text) > self.seq_length:
                return text[:self.seq_length]
            return np.pad(text, (0, self.seq_length - len(text)), 'constant', constant_values=0)

        name_tensors = []
        api_tensors = []
        token_tensors = []
        javadoc_tensors = []

        for i in range(0, self.data_count):
            padded_names = pad(self.vocabulary.get_id_or_unk_multiple(method_names[i]))
            name_tensors.append(padded_names)

            padded_api = pad(self.vocabulary.get_id_or_unk_multiple(method_api_calls[i]))
            api_tensors.append(padded_api)

            padded_tokens = pad(self.vocabulary.get_id_or_unk_multiple(method_tokens[i]))
            token_tensors.append(padded_tokens)

            padded_javadoc = pad(self.vocabulary.get_id_or_unk_multiple(javadoc[i]))
            javadoc_tensors.append(padded_javadoc)

        return np.array(name_tensors), np.array(api_tensors), \
               np.array(token_tensors), np.array(javadoc_tensors)


    def _load_data_file(self, file_name):
        dataset = []
        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()
                dataset.append(line.split())
        return dataset

    def _flatten(self, lists):
        flattened = []
        for token_list in lists:
            for lst in token_list:
                flattened += lst
        return flattened
