
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
        self.dense_units = 16
        self.batch_size = 128

        self.scope = "deep-cs"

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
            self.name_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name='name')
            self.api_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name='api')
            self.token_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name='token')
            self.javadoc_pos_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length],
                                                          name='javadoc-pos')
            self.javadoc_neg_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length],
                                                          name='javadoc-neg')

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.step_size)

            self._make_model()
    

    def train(self):

        with self._sess.graph.as_default():

            init_op = tf.variables_initializer(self._sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self._sess.run(init_op)


            javadoc_tensors_neg = np.copy(self.javadoc_tensors)
            np.random.shuffle(javadoc_tensors_neg)

            for index in range(0, self.data_count, self.batch_size):
                end = index + self.batch_size

                name_batch = self.name_tensors[index:end]
                api_batch = self.api_tensors[index:end]
                token_batch = self.token_tensors[index:end]
                javadoc_pos_batch = self.javadoc_tensors[index:end]
                javadoc_neg_batch = javadoc_tensors_neg[index:end]

                feed_dict = {
                    self.name_placeholder: name_batch,
                    self.api_placeholder: api_batch,
                    self.token_placeholder: token_batch,
                    self.javadoc_pos_placeholder: javadoc_pos_batch,
                    self.javadoc_neg_placeholder: javadoc_neg_batch
                }

                op_result = self._sess.run(self.loss_op, feed_dict=feed_dict)
                print(op_result)
                break



    def _make_model(self):

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # Method Name embedding
            name_emb, name_state = self._make_rnn_embedding(self.name_placeholder,
                                                           name="name-embedding")
            name_pooled = tf.layers.max_pooling1d(inputs=name_emb[1],
                                                  pool_size=(self.seq_length,),
                                                  strides=(1,),
                                                  name="name-pooling")
            name_pooled = tf.squeeze(name_pooled, axis=1)

            # API Embedding
            api_emb, api_state = self._make_rnn_embedding(self.api_placeholder,
                                                          name="api-embedding")
            api_pooled = tf.layers.max_pooling1d(inputs=api_emb[1],
                                                 pool_size=(self.seq_length,),
                                                 strides=(1,),
                                                 name="api-pooling")
            api_pooled = tf.squeeze(api_pooled, axis=1)

            # Fusion Layer
            code_concat = tf.concat([name_pooled, api_pooled], axis=1, name="code-concat")
            code_emb = tf.layers.dense(inputs=code_concat, units=self.dense_units,
                                       activation=tf.nn.tanh, name="fusion")

            # Javadoc Embeddings
            jd_pos_emb, jd_pos_state = self._make_rnn_embedding(self.javadoc_pos_placeholder,
                                                                name="jd-embedding")
            jd_pos_pooled = tf.layers.max_pooling1d(inputs=jd_pos_emb[1],
                                                    pool_size=(self.seq_length,),
                                                    strides=(1,),
                                                    name="jd-pooling")
            jd_pos_pooled = tf.squeeze(jd_pos_pooled, axis=1)

            jd_neg_emb, jd_neg_state = self._make_rnn_embedding(self.javadoc_neg_placeholder,
                                                                name="jd-embedding")
            jd_neg_pooled = tf.layers.max_pooling1d(inputs=jd_neg_emb[1],
                                                    pool_size=(self.seq_length,),
                                                    strides=(1,),
                                                    name="jd-pooling")
            jd_neg_pooled = tf.squeeze(jd_neg_pooled, axis=1)

            self.loss_op = tf.math.reduce_sum(
                    tf.constant(self.margin, dtype=tf.float32) - \
                    tf.losses.cosine_distance(labels=jd_pos_pooled, predictions=code_emb, axis=1) + \
                    tf.losses.cosine_distance(labels=jd_neg_pooled, predictions=code_emb, axis=1)
                )


    def _make_rnn_embedding(self, placeholder, name):
        one_hot = tf.one_hot(placeholder, depth=1, on_value=1.0, off_value=0.0, dtype=tf.float32)
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_units,
                                          activation=tf.nn.tanh)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_units,
                                          activation=tf.nn.tanh)
        emb, state = tf.nn.bidirectional_dynamic_rnn(
                                        cell_fw=cell_fw,
                                        cell_bw=cell_bw,
                                        inputs=one_hot,
                                        dtype=tf.float32,
                                        scope=name)
        return emb, state

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
