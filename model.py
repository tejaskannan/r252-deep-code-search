
import numpy as np
import tensorflow as tf
import gzip
import pickle
from os import mkdir
from os.path import exists

from datetime import datetime
from dpu_utils.mlutils import Vocabulary

from parser import JAVADOC_FILE_NAME, METHOD_NAME_FILE_NAME
from parser import METHOD_API_FILE_NAME, METHOD_TOKENS_FILE_NAME
from parameters import Parameters, params_from_dict
from dataset import Dataset, Batch
from utils import pad, lst_equal, log_record

LINE = "-" * 50

META_NAME = "meta.pkl.gz"
MODEL_NAME = "model.chk"

class Model:

    def __init__(self, params, train_dir="train_data/", valid_dir="validation_data/",
                       save_dir="trained_models/", log_dir="log/", model_path=None):
        self.params = params

        if train_dir[-1] != "/":
            train_dir += "/"
        self.train_dir = train_dir

        if valid_dir[-1] != "/":
            valid_dir += "/"
        self.valid_dir = valid_dir

        if save_dir[-1] != "/":
            save_dir += "/"
        self.save_dir = save_dir

        if log_dir[-1] != "/":
            log_dir += "/"
        self.log_dir = log_dir

        self.scope = "deep-cs"

        self.dataset = Dataset(train_dir=train_dir,
                               valid_dir=valid_dir,
                               max_seq_length=self.params.max_seq_length,
                               max_vocab_size=self.params.max_vocab_size)

        self._sess = tf.Session(graph=tf.Graph())

        with self._sess.graph.as_default():
            self.name_placeholder = tf.placeholder(dtype=tf.int32,
                                                   shape=[None, self.params.max_seq_length],
                                                   name="name")
            self.api_placeholder = tf.placeholder(dtype=tf.int32,
                                                  shape=[None, self.params.max_seq_length],
                                                  name="api")
            self.token_placeholder = tf.placeholder(dtype=tf.int32,
                                                    shape=[None, self.params.max_seq_length],
                                                    name="token")
            self.javadoc_pos_placeholder = tf.placeholder(dtype=tf.int32,
                                                          shape=[None, self.params.max_seq_length],
                                                          name="javadoc-pos")
            self.javadoc_neg_placeholder = tf.placeholder(dtype=tf.int32,
                                                          shape=[None, self.params.max_seq_length],
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
            log_record(csv_name, ["Epoch", "Avg Train Loss", "Avg Validation Loss"])

            for epoch in range(self.params.num_epochs):
                train_losses = []
                valid_losses = []

                train_batches = self.dataset.make_mini_batches(self.params.batch_size, train=True)
                valid_batches = self.dataset.make_mini_batches(self.params.batch_size, train=False)

                print(LINE)
                print("Epoch: {0}".format(epoch))
                print(LINE)

                num_train_batches = train_batches.num_batches
                num_valid_batches = valid_batches.num_batches

                for i in range(0, num_train_batches):
                    javadoc_neg, javadoc_neg_len = \
                            self._generate_neg_javadoc(train_batches.javadoc_batches[i],
                                                       train_batches.javadoc_len_batches[i])

                    feed_dict = {
                        self.name_placeholder: train_batches.name_batches[i],
                        self.api_placeholder: train_batches.api_batches[i],
                        self.token_placeholder: train_batches.token_batches[i],
                        self.javadoc_pos_placeholder: train_batches.javadoc_batches[i],
                        self.javadoc_neg_placeholder: javadoc_neg,
                        self.name_len_placeholder: train_batches.name_len_batches[i],
                        self.api_len_placehodler: train_batches.api_len_batches[i],
                        self.token_len_placeholder: train_batches.token_len_batches[i],
                        self.javadoc_pos_len_placeholder: train_batches.javadoc_len_batches[i],
                        self.javadoc_neg_len_placeholder: javadoc_neg_len
                    }

                    ops = [self.loss_op, self.optimizer_op]
                    op_result = self._sess.run(ops, feed_dict=feed_dict)

                    avg_train_loss = (op_result[0]) / self.params.batch_size
                    train_losses.append(avg_train_loss)

                    print("Training batch {0}/{1}: {2}".format(i, num_train_batches-1, avg_train_loss))

                for i in range(0, num_valid_batches):
                    javadoc_neg, javadoc_neg_len = \
                            self._generate_neg_javadoc(valid_batches.javadoc_batches[i],
                                                       valid_batches.javadoc_len_batches[i])

                    feed_dict = {
                        self.name_placeholder: valid_batches.name_batches[i],
                        self.api_placeholder: valid_batches.api_batches[i],
                        self.token_placeholder: valid_batches.token_batches[i],
                        self.javadoc_pos_placeholder: valid_batches.javadoc_batches[i],
                        self.javadoc_neg_placeholder: javadoc_neg,
                        self.name_len_placeholder: valid_batches.name_len_batches[i],
                        self.api_len_placehodler: valid_batches.api_len_batches[i],
                        self.token_len_placeholder: valid_batches.token_len_batches[i],
                        self.javadoc_pos_len_placeholder: valid_batches.javadoc_len_batches[i],
                        self.javadoc_neg_len_placeholder: javadoc_neg_len
                    }
                    valid_result = self._sess.run(self.loss_op, feed_dict=feed_dict)
                    avg_valid_loss = valid_result / self.params.batch_size
                    valid_losses.append(avg_valid_loss)

                avg_valid_loss = np.average(valid_losses)
                avg_train_loss = np.average(train_losses)

                log_record(csv_name, [str(epoch), str(avg_train_loss), str(avg_valid_loss)])

                print(LINE)
                print("Average training loss in epoch {0}: {1}".format(epoch, avg_train_loss))
                print("Average validation loss in epoch {0}: {1}".format(epoch, avg_valid_loss))

                if (avg_valid_loss < best_valid_loss):
                    best_valid_loss = avg_valid_loss
                    print("Saving model: " + train_name)
                    self.save(self.save_dir, train_name)

                print(LINE)

    def save(self, base_folder, name):
        meta_data = {
            "model_type": type(self).__name__,
            "parameters": self.params.as_dict(),
            "name": name
        }

        save_folder = base_folder + name
        if not exists(save_folder):
            mkdir(save_folder)

        meta_path = save_folder + "/" + META_NAME
        with gzip.GzipFile(meta_path, 'wb') as out_file:
            pickle.dump(meta_data, out_file)

        model_path = save_folder + "/" + MODEL_NAME
        saver = tf.train.Saver()
        saver.save(self._sess, model_path)

    # We assume that save_folder ends in a slash
    def restore(self, save_folder):
        meta_data = {}
        meta_path = save_folder + META_NAME
        with gzip.GzipFile(meta_path, 'rb') as in_file:
            meta_data = pickle.load(in_file)
        self.params = params_from_dict(meta_data["parameters"])

        with self._sess.graph.as_default():
            model_path = save_folder + MODEL_NAME
            saver = tf.train.Saver()
            saver.restore(self._sess, model_path)

    def embed_method(self, method_name, method_api, method_tokens):
        name_vec = self.dataset.vocabulary.get_id_or_unk_multiple(method_name)
        api_vec = self.dataset.vocabulary.get_id_or_unk_multiple(method_api)
        token_vec = self.dataset.vocabulary.get_id_or_unk_multiple(method_tokens)

        name_tensor = np.array([pad(name_vec, self.params.max_seq_length)])
        name_len_tensor = np.array([min(len(name_vec), self.params.max_seq_length)])
        api_tensor = np.array([pad(api_vec, self.params.max_seq_length)])
        api_len_tensor = np.array([min(len(api_vec), self.params.max_seq_length)])
        token_tensor = np.array([pad(token_vec, self.params.max_seq_length)])
        token_len_tensor = np.array([min(len(token_vec), self.params.max_seq_length)])

        with self._sess.graph.as_default():

            feed_dict = {
                self.name_placeholder: name_tensor,
                self.name_len_placeholder: name_len_tensor,
                self.api_placeholder: api_tensor,
                self.api_len_placehodler: api_len_tensor,
                self.token_placeholder: token_tensor,
                self.token_len_placeholder: token_len_tensor
            }

            embedding = self._sess.run(self.code_embedding, feed_dict=feed_dict)
        
        return embedding[0]

    def embed_description(self, description):
        descr_tokens = description.split(" ")
        descr_vec = self.dataset.vocabulary.get_id_or_unk_multiple(descr_tokens)

        descr_tensor = np.array([pad(descr_vec, self.params.max_seq_length)])
        descr_len_tensor = np.array([min(len(descr_vec), self.params.max_seq_length)])


        with self._sess.graph.as_default():
            feed_dict = {
                self.javadoc_pos_placeholder: descr_tensor,
                self.javadoc_pos_len_placeholder: descr_len_tensor
            }

            embedding = self._sess.run(self.description_embedding, feed_dict=feed_dict)
        return embedding[0]


    def _make_model(self):

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # Method Name embedding
            name_emb, name_state = self._make_rnn_embedding(self.name_placeholder,
                                                            self.name_len_placeholder,
                                                            name="name-embedding")
            # API Embedding
            api_emb, api_state = self._make_rnn_embedding(self.api_placeholder,
                                                          self.api_len_placehodler,
                                                          name="api-embedding")
            # Method Token embedding
            vocab_size = len(self.dataset.vocabulary)
            token_emb_var = tf.Variable(tf.random.uniform(shape=(vocab_size, self.params.embedding_size), maxval=1.0),
                                        name="token-embedding-var")
            token_emb = tf.nn.embedding_lookup(token_emb_var, self.token_placeholder)

            # We mask out elements which are padded before feeding tokens into max pooling
            index_list = tf.range(self.params.max_seq_length)
            index_tensor = tf.tile(tf.expand_dims(index_list, axis=0),
                                   multiples=(tf.shape(self.token_placeholder)[0],1))
            token_mask = index_tensor < tf.expand_dims(self.token_len_placeholder, axis=1)
            token_mask = tf.tile(tf.expand_dims(token_mask, axis=2),
                               multiples=(1,1,self.params.dense_units))

            token_emb *= tf.cast(token_mask, dtype=tf.float32)

            if self.params.combine_type == "attention":
                name_context = self._make_attention_layer(name_emb[1], name="name-attn")
                api_context = self._make_attention_layer(api_emb[1], name="api-attn")
                token_context = self._make_attention_layer(token_emb, name="token-attn")
            else:
                name_context = self._make_max_pooling_1d(token_emb[1], name="name-pooling")
                api_context = self._make_max_pooling_1d(api_emb[1], name="api-pooling")
                token_context = self._make_max_pooling_1d(token_emb, name="token-pooling")

            # Fusion Layer
            code_concat = tf.concat([name_context, api_context, token_context],
                                    axis=1, name="code-concat")
            code_emb = tf.layers.dense(inputs=code_concat, units=self.params.dense_units,
                                       activation=tf.nn.tanh, name="fusion")

            self.code_embedding = code_emb

            # Javadoc Embeddings
            jd_pos_emb, jd_pos_state = self._make_rnn_embedding(self.javadoc_pos_placeholder,
                                                                self.javadoc_pos_len_placeholder,
                                                                name="jd-embedding")

            jd_neg_emb, jd_neg_state = self._make_rnn_embedding(self.javadoc_neg_placeholder,
                                                                self.javadoc_neg_len_placeholder,
                                                                name="jd-embedding")

            if self.params.combine_type == "attention":
                jd_neg_context = self._make_attention_layer(jd_neg_emb[1], name="jd-attn")
                jd_pos_context = self._make_attention_layer(jd_pos_emb[1], name="jd-attn")
            else:
                jd_neg_context = self._make_max_pooling_1d(jd_neg_emb[1], name="jd-pooling")
                jd_pos_context = self._make_max_pooling_1d(jd_pos_emb[1], name="jd-pooling")

            self.description_embedding = jd_pos_context

            self.loss_op = tf.math.reduce_sum(
                    tf.constant(self.params.margin, dtype=tf.float32) - \
                    tf.losses.cosine_distance(labels=jd_pos_context, predictions=code_emb, axis=1) + \
                    tf.losses.cosine_distance(labels=jd_neg_context, predictions=code_emb, axis=1)
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
        vocab_size = len(self.dataset.vocabulary)
        encoding_var = tf.Variable(tf.random.uniform(shape=(vocab_size, self.params.embedding_size), maxval=1.0,
                                   name=name + "-var"))
        encoding = tf.nn.embedding_lookup(encoding_var, placeholder, name=name + "-enc")

        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.params.rnn_units,
                                          activation=tf.nn.tanh,
                                          name=name + "-fw")
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.params.rnn_units,
                                          activation=tf.nn.tanh,
                                          name=name + "-bw")

        initial_state_fw = cell_fw.zero_state(tf.shape(encoding)[0], dtype=tf.float32)
        initial_state_bw = cell_bw.zero_state(tf.shape(encoding)[0], dtype=tf.float32)
        emb, state = tf.nn.bidirectional_dynamic_rnn(
                                        cell_fw=cell_fw,
                                        cell_bw=cell_bw,
                                        inputs=encoding,
                                        sequence_length=len_placeholder,
                                        initial_state_fw=initial_state_fw,
                                        initial_state_bw=initial_state_bw,
                                        dtype=tf.float32,
                                        scope=name)
        return emb, state

    def _make_max_pooling_1d(self, inputs, name):
        pooled = tf.layers.max_pooling1d(inputs=inputs,
                                         pool_size=(self.params.max_seq_length,),
                                         strides=(1,),
                                         name=name)
        return tf.squeeze(pooled, axis=1)

    def _make_attention_layer(self, inputs, name):
        weights = tf.layers.dense(inputs=inputs, units=1, activation=tf.nn.tanh,
                                  name=name + "-attn-weights")
        alphas = tf.nn.softmax(weights, name=name + "-attn")
        return tf.reduce_sum(alphas * inputs, axis=1, name=name + "-attn-reduce")



    def _generate_neg_javadoc(self, javadoc, javadoc_len):
        neg_javadoc = []
        neg_javadoc_len = []
        for i, jd in enumerate(javadoc):
            rand_index = np.random.randint(0, len(javadoc))
            while lst_equal(javadoc[i], javadoc[rand_index]):
                rand_index = np.random.randint(0, len(javadoc))
            neg_javadoc.append(javadoc[rand_index])
            neg_javadoc_len.append(javadoc_len[rand_index])

        assert len(neg_javadoc) == len(javadoc)

        return neg_javadoc, neg_javadoc_len

