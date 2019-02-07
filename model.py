
import numpy as np
import tensorflow as tf
import gzip
import pickle
import csv

from datetime import datetime
from dpu_utils.mlutils import Vocabulary

from parser import JAVADOC_FILE_NAME, METHOD_NAME_FILE_NAME
from parser import METHOD_API_FILE_NAME, METHOD_TOKENS_FILE_NAME
from parameters import Parameters
from dataset import Dataset, Batch

LINE = "-" * 50

class Model:

    def __init__(self, train_dir="data", save_dir="trained_models/", log_dir="log/"):
        # Intialize some hyperparameters
        self.params = Parameters(
            train_frac = 0.8,
            valid_frac = 0.2,
            step_size = 0.01,
            gradient_clip = 1,
            margin = 0.05,
            max_vocab_size = 50000,
            seq_length = 50,
            rnn_units = 16,
            dense_units = 16,
            batch_size = 128,
            num_epochs = 2,
            optimizer="adam"
        )

        if save_dir[-1] != "/":
            save_dir += "/"
        self.save_dir = save_dir

        if log_dir[-1] != "/":
            log_dir += "/"
        self.log_dir = log_dir

        self.scope = "deep-cs"

        self.dataset = Dataset(data_dir=train_dir,
                               seq_length=self.params.seq_length,
                               max_vocab_size=self.params.max_vocab_size)

        self._sess = tf.Session(graph=tf.Graph())

        with self._sess.graph.as_default():
            self.name_placeholder = tf.placeholder(dtype=tf.int32,
                                                   shape=[None, self.params.seq_length],
                                                   name="name")
            self.api_placeholder = tf.placeholder(dtype=tf.int32,
                                                  shape=[None, self.params.seq_length],
                                                  name="api")
            self.token_placeholder = tf.placeholder(dtype=tf.int32,
                                                    shape=[None, self.params.seq_length],
                                                    name="token")
            self.javadoc_pos_placeholder = tf.placeholder(dtype=tf.int32,
                                                          shape=[None, self.params.seq_length],
                                                          name="javadoc-pos")
            self.javadoc_neg_placeholder = tf.placeholder(dtype=tf.int32,
                                                          shape=[None, self.params.seq_length],
                                                          name="javadoc-neg")
            self.name_len_placeholder = tf.placeholder(dtype=tf.int32,
                                                       shape=[None],
                                                       name="name-len")
            self.api_len_placehodler = tf.placeholder(dtype=tf.int32,
                                                      shape=[None],
                                                      name="api-len")
            self.token_len_placeholder = tf.placeholder(dtype=tf.int32,
                                                        shape=[None],
                                                        name="token-len")
            self.javadoc_pos_len_placeholder = tf.placeholder(dtype=tf.int32,
                                                              shape=[None],
                                                              name="jd-pos-len")
            self.javadoc_neg_len_placeholder = tf.placeholder(dtype=tf.int32,
                                                              shape=[None],
                                                              name="jd-neg-len")

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.params.step_size)

            self._make_model()
            self._make_training_step()
    

    def train(self):

        with self._sess.graph.as_default():

            init_op = tf.variables_initializer(self._sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self._sess.run(init_op)

            # Some large value
            best_valid_loss = 10000

            train_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            csv_name = self.log_dir + train_name + "-data.csv"
            self._log_record(csv_name, ["Epoch", "Avg Train Loss", "Avg Validation Loss"])

            for epoch in range(self.params.num_epochs):
                train_losses = []
                valid_losses = []

                batches = self.dataset.make_mini_batches(self.params.batch_size)

                print(LINE)
                print("Epoch: {0}".format(epoch))
                print(LINE)

                num_batches = batches.num_batches

                split_point = int(num_batches * self.params.train_frac)

                for i in range(0, split_point):
                    javadoc_neg, javadoc_neg_len = \
                            self._generate_neg_javadoc(batches.javadoc_batches[i],
                                                       batches.javadoc_len_batches[i])

                    feed_dict = {
                        self.name_placeholder: batches.name_batches[i],
                        self.api_placeholder: batches.api_batches[i],
                        self.token_placeholder: batches.token_batches[i],
                        self.javadoc_pos_placeholder: batches.javadoc_batches[i],
                        self.javadoc_neg_placeholder: javadoc_neg,
                        self.name_len_placeholder: batches.name_len_batches[i],
                        self.api_len_placehodler: batches.api_len_batches[i],
                        self.token_len_placeholder: batches.token_len_batches[i],
                        self.javadoc_pos_len_placeholder: batches.javadoc_len_batches[i],
                        self.javadoc_neg_len_placeholder: javadoc_neg_len
                    }

                    ops = [self.loss_op, self.optimizer_op]
                    op_result = self._sess.run(ops, feed_dict=feed_dict)
                    train_losses.append(op_result[0])

                    print("Training batch {0}/{1}: {2}".format(i, split_point-1, op_result[0]))

                for i in range(split_point, num_batches):
                    javadoc_neg, javadoc_neg_len = \
                            self._generate_neg_javadoc(batches.javadoc_batches[i],
                                                       batches.javadoc_len_batches[i])

                    feed_dict = {
                        self.name_placeholder: batches.name_batches[i],
                        self.api_placeholder: batches.api_batches[i],
                        self.token_placeholder: batches.token_batches[i],
                        self.javadoc_pos_placeholder: batches.javadoc_batches[i],
                        self.javadoc_neg_placeholder: javadoc_neg,
                        self.name_len_placeholder: batches.name_len_batches[i],
                        self.api_len_placehodler: batches.api_len_batches[i],
                        self.token_len_placeholder: batches.token_len_batches[i],
                        self.javadoc_pos_len_placeholder: batches.javadoc_len_batches[i],
                        self.javadoc_neg_len_placeholder: javadoc_neg_len
                    }
                    valid_result = self._sess.run(self.loss_op, feed_dict=feed_dict)
                    valid_losses.append(valid_result)

                avg_valid_loss = np.average(valid_losses)
                avg_train_loss = np.average(train_losses)

                self._log_record(csv_name, [str(epoch), str(avg_train_loss), str(avg_valid_loss)])

                print(LINE)
                print("Average training loss in epoch {0}: {1}".format(epoch, avg_train_loss))
                print("Average validation loss in epoch {0}: {1}".format(epoch, avg_valid_loss))

                if (avg_valid_loss < best_valid_loss):
                    best_valid_loss = avg_valid_loss
                    name = train_name + "-" + str(epoch)
                    path = self.save_dir + name + "_best.pkl.gz"
                    print("Saving model: " + name)
                    self.save(path, name)

                print(LINE)

    def save(self, path, name):
        variables_to_save = list(set(self._sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        weights_to_save = self._sess.run(variables_to_save)
        weights_dict = {
            var.name: value for (var, value) in zip(variables_to_save, weights_to_save)
        }
        data = {
            "model_type": type(self).__name__,
            "parameters": self.params.as_dict(),
            "weights": weights_dict,
            "name": name
        }

        with gzip.GzipFile(path, 'wb') as out_file:
            pickle.dump(data, out_file)


    def _make_model(self):

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # Method Name embedding
            name_emb, name_state = self._make_rnn_embedding(self.name_placeholder,
                                                            self.name_len_placeholder,
                                                            name="name-embedding")
            name_pooled = self._make_max_pooling_1d(name_emb[1], name="name-pooling")

            # API Embedding
            api_emb, api_state = self._make_rnn_embedding(self.api_placeholder,
                                                          self.api_len_placehodler,
                                                          name="api-embedding")
            api_pooled = self._make_max_pooling_1d(api_emb[1], name="api-pooling")

            # Method Token embedding
            one_hot_tokens = tf.one_hot(self.token_placeholder,
                                       depth=len(self.dataset.vocabulary),
                                       on_value=1.0,
                                       off_value=0.0,
                                       dtype=tf.float32)
            token_emb = tf.layers.dense(inputs=one_hot_tokens,
                                        units=self.params.dense_units,
                                        activation=tf.nn.tanh)

            # We mask out elements which are padded before feeding tokens into max pooling
            index_list = tf.range(self.params.seq_length)
            index_tensor = tf.tile(tf.expand_dims(index_list, axis=0),
                                   multiples=(tf.shape(self.token_placeholder)[0],1))

            token_mask = index_tensor < tf.expand_dims(self.token_len_placeholder, axis=1)
            exp_mask = tf.tile(tf.expand_dims(token_mask, axis=2),
                               multiples=(1,1,self.params.dense_units))

            token_emb *= tf.cast(exp_mask, dtype=tf.float32)
            token_pooled = self._make_max_pooling_1d(token_emb, name="token-pooling")

            # Fusion Layer
            code_concat = tf.concat([name_pooled, api_pooled, token_pooled],
                                    axis=1, name="code-concat")
            code_emb = tf.layers.dense(inputs=code_concat, units=self.params.dense_units,
                                       activation=tf.nn.tanh, name="fusion")

            # Javadoc Embeddings
            jd_pos_emb, jd_pos_state = self._make_rnn_embedding(self.javadoc_pos_placeholder,
                                                                self.javadoc_pos_len_placeholder,
                                                                name="jd-embedding")
            jd_pos_pooled = self._make_max_pooling_1d(jd_pos_emb[1], name="jd-pooling")

            jd_neg_emb, jd_neg_state = self._make_rnn_embedding(self.javadoc_neg_placeholder,
                                                                self.javadoc_neg_len_placeholder,
                                                                name="jd-embedding")
            jd_neg_pooled = self._make_max_pooling_1d(jd_neg_emb[1], name="jd-pooling")

            self.loss_op = tf.math.reduce_sum(
                    tf.constant(self.params.margin, dtype=tf.float32) - \
                    tf.losses.cosine_distance(labels=jd_pos_pooled, predictions=code_emb, axis=1) + \
                    tf.losses.cosine_distance(labels=jd_neg_pooled, predictions=code_emb, axis=1)
                )

    def _make_training_step(self):
        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.loss_op, trainable_vars)
        clipped_grad, _ = tf.clip_by_global_norm(gradients, self.params.gradient_clip)
        pruned_gradients = []
        for grad, var in zip(clipped_grad, trainable_vars):
            if grad != None:
                pruned_gradients.append((grad, var))

        self.optimizer_op = self.optimizer.apply_gradients(pruned_gradients)


    def _make_rnn_embedding(self, placeholder, len_placeholder, name):
        one_hot = tf.one_hot(placeholder,
                             depth=len(self.dataset.vocabulary),
                             on_value=1.0,
                             off_value=0.0,
                             dtype=tf.float32)
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.params.rnn_units,
                                          activation=tf.nn.tanh)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.params.rnn_units,
                                          activation=tf.nn.tanh)
        emb, state = tf.nn.bidirectional_dynamic_rnn(
                                        cell_fw=cell_fw,
                                        cell_bw=cell_bw,
                                        inputs=one_hot,
                                        sequence_length=len_placeholder,
                                        dtype=tf.float32,
                                        scope=name)
        return emb, state

    def _make_max_pooling_1d(self, inpt, name):
        pooled = tf.layers.max_pooling1d(inputs=inpt,
                                         pool_size=(self.params.seq_length,),
                                         strides=(1,),
                                         name=name)
        return tf.squeeze(pooled, axis=1)

    def _generate_neg_javadoc(self, javadoc, javadoc_len):
        neg_javadoc = []
        neg_javadoc_len = []
        for i, jd in enumerate(javadoc):
            rand_index = np.random.randint(0, len(javadoc))
            while self._lst_equal(javadoc[i], javadoc[rand_index]):
                rand_index = np.random.randint(0, len(javadoc))
            neg_javadoc.append(javadoc[rand_index])
            neg_javadoc_len.append(javadoc_len[rand_index])

        assert len(neg_javadoc) == len(javadoc)

        return neg_javadoc, neg_javadoc_len


    def _lst_equal(_, lst1, lst2):
        if len(lst1) != len(lst2):
            return False
        for i in range(0, len(lst1)):
            if (lst1[i] != lst2[i]):
                return False
        return True

    def _log_record(self, file_name, record):
        with open(file_name, "a+") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",", quotechar="|")
            csv_writer.writerow(record)
