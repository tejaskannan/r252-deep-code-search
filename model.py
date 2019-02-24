
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

BIG_NUMBER = 1e7

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
            self.api_len_placeholder = tf.placeholder(dtype=tf.int32,
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

            train_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            train_name = self.params.combine_type + "-" + self.params.seq_embedding + "-" + train_time
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
                    neg_batch_index = np.random.randint(0, num_train_batches)
                    javadoc_neg_batch, javadoc_neg_len_batch = \
                            self._generate_neg_javadoc(train_batches.javadoc_batches[neg_batch_index],
                                                       train_batches.javadoc_len_batches[neg_batch_index])

                    feed_dict = {
                        self.name_placeholder: train_batches.name_batches[i],
                        self.api_placeholder: train_batches.api_batches[i],
                        self.token_placeholder: train_batches.token_batches[i],
                        self.javadoc_pos_placeholder: train_batches.javadoc_batches[i],
                        self.javadoc_neg_placeholder: train_batches.javadoc_batches[i],
                        self.name_len_placeholder: train_batches.name_len_batches[i],
                        self.api_len_placeholder: train_batches.api_len_batches[i],
                        self.token_len_placeholder: train_batches.token_len_batches[i],
                        self.javadoc_pos_len_placeholder: train_batches.javadoc_len_batches[i],
                        self.javadoc_neg_len_placeholder: train_batches.javadoc_len_batches[i]
                    }

                    ops = [self.loss_op, self.description_embedding, self.neg_descr_embedding, self.code_embedding, self.optimizer_op]
                    op_result = self._sess.run(ops, feed_dict=feed_dict)

                    avg_train_loss = (op_result[0]) / self.params.batch_size
                    train_losses.append(avg_train_loss)

                    print("Training batch {0}/{1}: {2}".format(i, num_train_batches-1, avg_train_loss))
                    break

                break

                for i in range(0, num_valid_batches):
                    neg_batch_index = np.random.randint(0, num_valid_batches)
                    javadoc_neg, javadoc_neg_len = \
                            self._generate_neg_javadoc(valid_batches.javadoc_batches[neg_batch_index],
                                                       valid_batches.javadoc_len_batches[neg_batch_index])

                    feed_dict = {
                        self.name_placeholder: valid_batches.name_batches[i],
                        self.api_placeholder: valid_batches.api_batches[i],
                        self.token_placeholder: valid_batches.token_batches[i],
                        self.javadoc_pos_placeholder: valid_batches.javadoc_batches[i],
                        self.javadoc_neg_placeholder: javadoc_neg,
                        self.name_len_placeholder: valid_batches.name_len_batches[i],
                        self.api_len_placeholder: valid_batches.api_len_batches[i],
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
            "name": name,
            "vocabulary": self.dataset.vocabulary
        }

        save_folder = base_folder + name
        if not exists(save_folder):
            mkdir(save_folder)

        meta_path = save_folder + "/" + META_NAME
        with gzip.GzipFile(meta_path, "wb") as out_file:
            pickle.dump(meta_data, out_file)

        model_path = save_folder + "/" + MODEL_NAME
        saver = tf.train.Saver()
        saver.save(self._sess, model_path)

    # We assume that save_folder ends in a slash
    def restore(self, save_folder):
        meta_data = {}
        meta_path = save_folder + META_NAME
        with gzip.GzipFile(meta_path, "rb") as in_file:
            meta_data = pickle.load(in_file)
        self.params = params_from_dict(meta_data["parameters"])
        self.dataset.vocabulary = meta_data["vocabulary"]

        with self._sess.graph.as_default():
            model_path = save_folder + MODEL_NAME
            saver = tf.train.Saver()
            saver.restore(self._sess, model_path)

    def embed_method(self, method_name, method_api, method_tokens):

        name_tok = method_name.split(" ")
        api_tok = method_api.split(" ")
        method_tok = method_tokens.split(" ")

        name_vec = self.dataset.vocabulary.get_id_or_unk_multiple(name_tok,
                                                                  pad_to_size=self.params.max_seq_length)
        api_vec = self.dataset.vocabulary.get_id_or_unk_multiple(api_tok,
                                                                 pad_to_size=self.params.max_seq_length)
        token_vec = self.dataset.vocabulary.get_id_or_unk_multiple(method_tok,
                                                                   pad_to_size=self.params.max_seq_length)

        name_tensor = np.array(name_vec)
        name_len_tensor = np.array([min(len(name_tok), self.params.max_seq_length)])
        api_tensor = np.array(api_vec)
        api_len_tensor = np.array([min(len(api_tok), self.params.max_seq_length)])
        token_tensor = np.array(token_vec)
        token_len_tensor = np.array([min(len(method_tok), self.params.max_seq_length)])

        with self._sess.graph.as_default():

            feed_dict = {
                self.name_placeholder: name_tensor,
                self.name_len_placeholder: name_len_tensor,
                self.api_placeholder: api_tensor,
                self.api_len_placeholder: api_len_tensor,
                self.token_placeholder: token_tensor,
                self.token_len_placeholder: token_len_tensor
            }

            embedding = self._sess.run(self.code_embedding, feed_dict=feed_dict)
        
        return embedding[0]

    def embed_description(self, description):
        descr_tokens = description.split(" ")
        descr_vec = self.dataset.vocabulary.get_id_or_unk_multiple(descr_tokens,
                                                                   pad_to_size=self.params.max_seq_length)

        descr_tensor = np.array(descr_vec)
        descr_len_tensor = np.array([min(len(descr_tokens), self.params.max_seq_length)])

        with self._sess.graph.as_default():
            feed_dict = {
                self.javadoc_pos_placeholder: descr_tensor,
                self.javadoc_pos_len_placeholder: descr_len_tensor
            }

            embedding = self._sess.run(self.description_embedding, feed_dict=feed_dict)

        return embedding[0]


    def _make_model(self):

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):


            if self.params.seq_embedding == "conv":
                name_embedding = self._conv_1d_embedding(self.name_placeholder,
                                                         self.name_len_placeholder,
                                                         name="name-embedding")
                api_embedding = self._conv_1d_embedding(self.api_placeholder,
                                                        self.api_len_placeholder,
                                                        name="api-embedding")
            else:
                # Method Name embedding
                name_emb, name_state = self._rnn_embedding(self.name_placeholder,
                                                           self.name_len_placeholder,
                                                           name="name-embedding")
                # API Embedding
                api_emb, api_state = self._rnn_embedding(self.api_placeholder,
                                                         self.api_len_placeholder,
                                                         name="api-embedding")

                # We combine the outputs from the forward and backward passes
                name_embedding = self._reduction_layer(name_emb,
                                                       self.params.embedding_size,
                                                       self.name_len_placeholder,
                                                       name="name-reduction")
                api_embedding = self._reduction_layer(api_emb,
                                                      self.params.embedding_size,
                                                      self.api_len_placeholder,
                                                      name="api-reduction")
            
            # Method Token embedding
            vocab_size = len(self.dataset.vocabulary)
            token_emb_var = tf.Variable(tf.random.normal(shape=(vocab_size, self.params.embedding_size)),
                                        name="token-embedding-var")
            token_embedding = tf.nn.embedding_lookup(token_emb_var, self.token_placeholder)
            token_embedding = tf.layers.dense(inputs=token_embedding,
                                              units=self.params.embedding_size,
                                              activation=tf.nn.tanh,
                                              name="token-embedding-dense")

            token_mask = self._create_mask(token_embedding, self.token_len_placeholder)
            token_embedding += token_mask

            if self.params.combine_type == "attention":
                name_context = self._attention_layer(name_embedding, name="name-attn")
                api_context = self._attention_layer(api_embedding, name="api-attn")
                token_context = self._attention_layer(token_embedding, name="token-attn")
            else:
                # Max Pooling
                name_context = tf.reduce_max(name_embedding, axis=1, name="name-pooling")
                api_context = tf.reduce_max(api_embedding, axis=1, name="api-pooling")
                token_context = tf.reduce_max(token_embedding, axis=1, name="token-pooling")

            self.name_context = name_context
            self.api_context = api_context
            self.token_context = token_context

            # Fusion Layer
            code_concat = tf.concat([name_context, api_context, token_context],
                                    axis=1, name="code-concat")
            fusion_hidden = tf.layers.dense(inputs=code_concat,
                                            units=self.params.hidden_fusion_units,
                                            activation=tf.nn.tanh,
                                            name="code-function-hidden")

            self.code_embedding = tf.layers.dense(inputs=fusion_hidden,
                                                  units=self.params.embedding_size,
                                                  activation=tf.nn.tanh,
                                                  name="code-fusion")

            # Javadoc Embeddings
            if self.params.seq_embedding == "conv":
                jd_pos_embedding = self._conv_1d_embedding(self.javadoc_pos_placeholder,
                                                           self.javadoc_pos_len_placeholder,
                                                           name="jd-embedding")
                jd_neg_embedding = self._conv_1d_embedding(self.javadoc_neg_placeholder,
                                                           self.javadoc_neg_len_placeholder,
                                                           name="jd-embedding")
            else:
                jd_pos_emb, jd_pos_state = self._rnn_embedding(self.javadoc_pos_placeholder,
                                                               self.javadoc_pos_len_placeholder,
                                                               name="jd-embedding")

                jd_neg_emb, jd_neg_state = self._rnn_embedding(self.javadoc_neg_placeholder,
                                                               self.javadoc_neg_len_placeholder,
                                                               name="jd-embedding")

                jd_pos_embedding = self._reduction_layer(jd_pos_emb,
                                                         self.params.embedding_size,
                                                         self.javadoc_pos_len_placeholder,
                                                         name="jd-reduction")
                jd_neg_embedding = self._reduction_layer(jd_neg_emb,
                                                         self.params.embedding_size,
                                                         self.javadoc_neg_len_placeholder,
                                                         name="jd-reduction")

            if self.params.combine_type == "attention":
                jd_neg_context = self._attention_layer(jd_neg_embedding, name="jd-attn")
                jd_pos_context = self._attention_layer(jd_pos_embedding, name="jd-attn")
            else:
                jd_neg_context = tf.reduce_max(jd_neg_embedding, axis=1, name="jd-pooling")
                jd_pos_context = tf.reduce_max(jd_pos_embedding, axis=1, name="jd_pooling")

            self.description_embedding = jd_pos_context
            self.neg_descr_embedding = jd_neg_context

            code_embedding = tf.math.l2_normalize(self.code_embedding, axis=1)
            jd_pos_context = tf.math.l2_normalize(jd_pos_context, axis=1)
            jd_neg_context = tf.math.l2_normalize(jd_neg_context, axis=1)


            self.loss_op = tf.reduce_sum(tf.nn.relu(
                tf.constant(self.params.margin, dtype=tf.float32) + \
                    tf.losses.cosine_distance(jd_pos_context, code_embedding, axis=1) - \
                    tf.losses.cosine_distance(jd_neg_context, code_embedding, axis=1)
            ))


    def _make_training_step(self):
        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.loss_op, trainable_vars)
        clipped_grad, _ = tf.clip_by_global_norm(gradients, self.params.gradient_clip)
        pruned_gradients = []
        for grad, var in zip(clipped_grad, trainable_vars):
            if grad != None:
                pruned_gradients.append((grad, var))

        self.optimizer_op = self.optimizer.apply_gradients(pruned_gradients)


    def _rnn_embedding(self, placeholder, len_placeholder, name):
        vocab_size = len(self.dataset.vocabulary)
        encoding_var = tf.Variable(tf.random.normal(shape=(vocab_size, self.params.embedding_size)),
                                   name=name + "-var")
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
                                        initial_state_fw=initial_state_fw,
                                        initial_state_bw=initial_state_bw,
                                        dtype=tf.float32,
                                        scope=name)
        return emb, state

    def _conv_1d_embedding(self, placeholder, len_placeholder, name):
        vocab_size = len(self.dataset.vocabulary)
        encoding_var = tf.Variable(tf.random.normal(shape=(vocab_size, self.params.embedding_size)),
                                   name=name + "-var")
        encoding = tf.nn.embedding_lookup(encoding_var, placeholder, name=name + "-enc")

        embedding = tf.layers.conv1d(inputs=encoding,
                                     filters=self.params.embedding_size,
                                     kernel_size=self.params.kernel_size,
                                     padding="same",
                                     activation=tf.nn.tanh,
                                     name=name + "-conv-emb")

        # Mask the output to adjust for variable sequence lengths
        mask = self._create_mask(placeholder, len_placeholder)
        return embedding + mask

    def _max_pooling_1d(self, inputs, name):
        pooled = tf.layers.max_pooling1d(inputs=inputs,
                                         pool_size=(self.params.max_seq_length,),
                                         strides=(1,),
                                         name=name)
        return tf.squeeze(pooled, axis=1)

    def _attention_layer(self, inputs, name):
        weights = tf.layers.dense(inputs=inputs, units=1, activation=tf.nn.tanh,
                                  name=name + "-attn-weights")
        alphas = tf.nn.softmax(weights, name=name + "-attn")
        return tf.reduce_sum(alphas * inputs, axis=1, name=name + "-attn-reduce")

    def _reduction_layer(self, rnn_embedding, output_size, len_placeholder, name):
        concat_tensor = tf.concat([rnn_embedding[0], rnn_embedding[1]], axis=2,
                                  name=name + "-concat")
        reduction = tf.layers.dense(inputs=concat_tensor,
                                    units=output_size,
                                    use_bias=False,
                                    name=name + "-dense")

        mask = self._create_mask(reduction, len_placeholder)

        return reduction + mask

    def _create_mask(self, placeholder, len_placeholder):
        # We mask out elements which are padded before feeding tokens into an aggregation layer
        index_list = tf.range(self.params.max_seq_length)  # S
        index_tensor = tf.tile(tf.expand_dims(index_list, axis=0),
                               multiples=(tf.shape(placeholder)[0],1))  # B x S

        mask = index_tensor < tf.expand_dims(len_placeholder, axis=1)  # B x S

        mask = tf.tile(tf.expand_dims(mask, axis=2),  # B x S x E
                      multiples=(1,1,self.params.embedding_size))

        return (1 - tf.cast(mask, dtype=tf.float32)) * -BIG_NUMBER

    def _cosine_similarity(self, labels, predictions):
        dot_prod = tf.reduce_sum(labels * predictions, axis=1)
        label_norm = tf.norm(labels, axis=1)
        predict_norm = tf.norm(predictions, axis=1)
        return dot_prod / (label_norm * predict_norm)

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

        for i in range(len(javadoc)):
            assert not lst_equal(javadoc[i], neg_javadoc[i])

        return neg_javadoc, neg_javadoc_len

