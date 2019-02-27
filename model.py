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
from utils import lst_equal, log_record, add_slash_to_end
from constants import *


class Model:

    def __init__(self, params, train_dir='train_data/', valid_dir='validation_data/',
                 save_dir='trained_models/', log_dir='log/'):
        self.params = params

        self.train_dir = add_slash_to_end(train_dir)
        self.valid_dir = add_slash_to_end(valid_dir)
        self.save_dir = add_slash_to_end(save_dir)
        self.log_dir = add_slash_to_end(log_dir)

        self.scope = 'deep-cs'

        max_seq_len = self.params.max_seq_length

        self.dataset = Dataset(train_dir=train_dir,
                               valid_dir=valid_dir,
                               max_seq_length=max_seq_len,
                               max_vocab_size=self.params.max_vocab_size)

        self._sess = tf.Session(graph=tf.Graph())

        with self._sess.graph.as_default():
            # Placeholders for Token Sequences
            self.method_names = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len], name='names')
            self.method_apis = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len], name='apis')
            self.method_tokens = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len], name='tokens')
            self.description = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len], name='descr')
            self.description_neg = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len], name='descr-neg')

            # Placeholders for Token Sequence Lengths
            self.method_names_len = tf.placeholder(dtype=tf.int32, shape=[None], name='names-len')
            self.method_apis_len = tf.placeholder(dtype=tf.int32, shape=[None], name='apis-len')
            self.method_tokens_len = tf.placeholder(dtype=tf.int32, shape=[None], name='tokens-len')
            self.description_len = tf.placeholder(dtype=tf.int32, shape=[None], name='descr-len')
            self.description_neg_len = tf.placeholder(dtype=tf.int32, shape=[None], name='descr-neg-len')

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.params.step_size)

            self._make_model()
            self._make_training_step()

    def train(self):

        with self._sess.graph.as_default():

            init_op = tf.variables_initializer(self._sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self._sess.run(init_op)

            # Initializes a name for this training run using the current time
            train_time = datetime.now().strftime(DATE_FORMAT)
            train_name = NAME_FORMAT.format(self.params.combine_type, self.params.seq_embedding, train_time)

            # Initalize logging of this training run
            csv_name = LOG_FORMAT.format(self.log_dir, train_name)
            log_record(csv_name, ['Epoch', 'Avg Train Loss', 'Avg Validation Loss'])

            # Initialize best loss to a large value
            best_valid_loss = BIG_NUMBER

            for epoch in range(self.params.num_epochs):
                train_losses = []
                valid_losses = []

                train_batches = self.dataset.make_mini_batches(self.params.batch_size, train=True)
                valid_batches = self.dataset.make_mini_batches(self.params.batch_size, train=False)

                print(LINE)
                print('Epoch: {0}'.format(epoch))
                print(LINE)

                num_train_batches = train_batches.num_batches
                num_valid_batches = valid_batches.num_batches

                for i in range(num_train_batches):
                    javadoc_neg_batch, javadoc_neg_len_batch = \
                            self._generate_neg_javadoc(train_batches.javadoc_batches[i],
                                                       train_batches.javadoc_len_batches[i])

                    feed_dict = {
                        self.method_names: train_batches.name_batches[i],
                        self.method_apis: train_batches.api_batches[i],
                        self.method_tokens: train_batches.token_batches[i],
                        self.description: train_batches.javadoc_batches[i],
                        self.description_neg: javadoc_neg_batch,
                        self.method_names_len: train_batches.name_len_batches[i],
                        self.method_apis_len: train_batches.api_len_batches[i],
                        self.method_tokens_len: train_batches.token_len_batches[i],
                        self.description_len: train_batches.javadoc_len_batches[i],
                        self.description_neg_len: javadoc_neg_len_batch
                    }

                    ops = [self.loss_op, self.optimizer_op]
                    op_result = self._sess.run(ops, feed_dict=feed_dict)

                    avg_train_loss = (op_result[0]) / self.params.batch_size
                    train_losses.append(avg_train_loss)

                    print('Training batch {0}/{1}: {2}'.format(i, num_train_batches-1, avg_train_loss))

                print(LINE)

                for i in range(num_valid_batches):
                    javadoc_neg_batch, javadoc_neg_len_batch = \
                            self._generate_neg_javadoc(valid_batches.javadoc_batches[i],
                                                       valid_batches.javadoc_len_batches[i])

                    feed_dict = {
                        self.method_names: valid_batches.name_batches[i],
                        self.method_apis: valid_batches.api_batches[i],
                        self.method_tokens: valid_batches.token_batches[i],
                        self.description: valid_batches.javadoc_batches[i],
                        self.description_neg: javadoc_neg_batch,
                        self.method_names_len: valid_batches.name_len_batches[i],
                        self.method_apis_len: valid_batches.api_len_batches[i],
                        self.method_tokens_len: valid_batches.token_len_batches[i],
                        self.description_len: valid_batches.javadoc_len_batches[i],
                        self.description_neg_len: javadoc_neg_len_batch
                    }
                    valid_result = self._sess.run(self.loss_op, feed_dict=feed_dict)
                    avg_valid_loss = valid_result / self.params.batch_size
                    valid_losses.append(avg_valid_loss)

                    print('Validation batch {0}/{1}: {2}'.format(i, num_valid_batches-1, avg_valid_loss))

                avg_valid_loss = np.average(valid_losses)
                avg_train_loss = np.average(train_losses)

                log_record(csv_name, [str(epoch), str(avg_train_loss), str(avg_valid_loss)])

                print(LINE)
                print('Average training loss in epoch {0}: {1}'.format(epoch, avg_train_loss))
                print('Average validation loss in epoch {0}: {1}'.format(epoch, avg_valid_loss))

                if (avg_valid_loss < best_valid_loss):
                    best_valid_loss = avg_valid_loss
                    print('Saving model: ' + train_name)
                    self.save(self.save_dir, train_name)

                print(LINE)

    def save(self, base_folder, name):
        meta_data = {
            'model_type': type(self).__name__,
            'parameters': self.params.as_dict(),
            'name': name,
            'vocabulary': self.dataset.vocabulary
        }

        save_folder = base_folder + name
        if not exists(save_folder):
            mkdir(save_folder)

        meta_path = save_folder + '/' + META_NAME
        with gzip.GzipFile(meta_path, 'wb') as out_file:
            pickle.dump(meta_data, out_file)

        model_path = save_folder + '/' + MODEL_NAME
        saver = tf.train.Saver()
        saver.save(self._sess, model_path)

    # We assume that save_folder ends in a slash
    def restore(self, save_folder):
        meta_data = {}
        meta_path = save_folder + META_NAME
        with gzip.GzipFile(meta_path, 'rb') as in_file:
            meta_data = pickle.load(in_file)
        self.params = params_from_dict(meta_data['parameters'])
        self.dataset.vocabulary = meta_data['vocabulary']

        with self._sess.graph.as_default():
            model_path = save_folder + MODEL_NAME
            saver = tf.train.Saver()
            saver.restore(self._sess, model_path)

    def embed_method(self, method_name, method_api, method_tokens):

        name_tok = method_name.split(' ')
        api_tok = method_api.split(' ')
        method_tok = method_tokens.split(' ')

        name_vec = self.dataset.vocabulary.get_id_or_unk_multiple(name_tok,
                                                                  pad_to_size=self.params.max_seq_length)
        api_vec = self.dataset.vocabulary.get_id_or_unk_multiple(api_tok,
                                                                 pad_to_size=self.params.max_seq_length)
        token_vec = self.dataset.vocabulary.get_id_or_unk_multiple(method_tok,
                                                                   pad_to_size=self.params.max_seq_length)

        name_tensor = np.array([name_vec])
        name_len_tensor = np.array([min(len(name_tok), self.params.max_seq_length)])
        api_tensor = np.array([api_vec])
        api_len_tensor = np.array([min(len(api_tok), self.params.max_seq_length)])
        token_tensor = np.array([token_vec])
        token_len_tensor = np.array([min(len(method_tok), self.params.max_seq_length)])

        with self._sess.graph.as_default():

            feed_dict = {
                self.method_names: name_tensor,
                self.method_names_len: name_len_tensor,
                self.method_apis: api_tensor,
                self.method_apis_len: api_len_tensor,
                self.method_tokens: token_tensor,
                self.method_tokens_len: token_len_tensor
            }

            embedding = self._sess.run(self.code_embedding, feed_dict=feed_dict)

        return embedding[0]

    def embed_description(self, description):
        descr_tokens = description.split(' ')
        descr_vec = self.dataset.vocabulary.get_id_or_unk_multiple(descr_tokens,
                                                                   pad_to_size=self.params.max_seq_length)

        descr_tensor = np.array([descr_vec])
        descr_len_tensor = np.array([min(len(descr_tokens), self.params.max_seq_length)])

        with self._sess.graph.as_default():
            feed_dict = {
                self.description: descr_tensor,
                self.description_len: descr_len_tensor
            }

            embedding = self._sess.run(self.descr_embedding, feed_dict=feed_dict)

        return embedding[0]

    def _make_model(self):

        vocab_size = len(self.dataset.vocabulary)
        embed_size = self.params.embedding_size

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

            # Encode input tensors
            name_enc_var = tf.Variable(tf.random.normal(shape=(vocab_size, embed_size)), name='name-enc-var')
            name_encoding = tf.nn.embedding_lookup(name_enc_var, self.method_names)

            api_enc_var = tf.Variable(tf.random.normal(shape=(vocab_size, embed_size)), name='api-enc-var')
            api_encoding = tf.nn.embedding_lookup(api_enc_var, self.method_apis)

            token_enc_var = tf.Variable(tf.random.normal(shape=(vocab_size, embed_size)), name='token-enc-var')
            token_encoding = tf.nn.embedding_lookup(token_enc_var, self.method_tokens)

            descr_enc_var = tf.Variable(tf.random.normal(shape=(vocab_size, embed_size)), name='descr-enc-var')
            descr_encoding = tf.nn.embedding_lookup(descr_enc_var, self.description, name='descr-enc')
            descr_neg_encoding = tf.nn.embedding_lookup(descr_enc_var, self.description_neg, name='descr-enc')

            # Method Token embedding
            token_embedding = tf.layers.dense(inputs=token_encoding,
                                              units=self.params.embedding_size,
                                              activation=tf.nn.tanh,
                                              name='token-embedding')
            token_mask = self._create_mask(token_embedding, self.method_tokens_len)
            token_embedding += token_mask

            # Embed sequences. By default we use a BiRNN.
            if self.params.seq_embedding == 'conv':
                name_embedding = self._conv_1d_embedding(name_encoding, self.method_names_len,
                                                         name='name-embed')
                api_embedding = self._conv_1d_embedding(api_encoding, self.method_apis_len,
                                                        name='api-embed')
                descr_embedding = self._conv_1d_embedding(descr_encoding, self.description_len,
                                                          name='descr-embed')
                descr_neg_embedding = self._conv_1d_embedding(descr_neg_encoding, self.description_neg_len,
                                                              name='descr-neg-embed')
            else:
                # Embeddings using a BiRNN
                name_emb, _name_state = self._rnn_embedding(name_encoding,
                                                            self.method_names_len,
                                                            name='name-rnn')
                api_emb, _api_state = self._rnn_embedding(api_encoding,
                                                          self.method_apis_len,
                                                          name='api-rnn')

                descr_emb, _descr_state = self._rnn_embedding(descr_encoding,
                                                              self.description_len,
                                                              name='descr-rnn')

                descr_neg_emb, _descr_neg_state = self._rnn_embedding(descr_neg_encoding,
                                                                      self.description_neg_len,
                                                                      name='descr-rnn')

                # Combine the outputs from the forward and backward passes
                name_embedding = self._reduction_layer(name_emb, embed_size, self.method_names_len,
                                                       name='name-embed')
                api_embedding = self._reduction_layer(api_emb, embed_size, self.method_apis_len,
                                                      name='api-embed')
                descr_embedding = self._reduction_layer(descr_emb, embed_size, self.description_len,
                                                        name='descr-embed')
                descr_neg_embedding = self._reduction_layer(descr_neg_emb, embed_size, self.description_neg_len,
                                                            name='descr-embed')

            # Combine sequences into a single context vector. By default we use max pooling.
            if self.params.combine_type == 'attention':
                name_context = self._attention_layer(name_embedding, name='name-attn')
                api_context = self._attention_layer(api_embedding, name='api-attn')
                token_context = self._attention_layer(token_embedding, name='token-attn')
                descr_context = self._attention_layer(descr_embedding, name='descr-attn')
                descr_neg_context = self._attention_layer(descr_neg_embedding, name='descr-attn')
            else:
                # Max Pooling
                name_context = tf.reduce_max(name_embedding, axis=1, name='name-pooling')
                api_context = tf.reduce_max(api_embedding, axis=1, name='api-pooling')
                token_context = tf.reduce_max(token_embedding, axis=1, name='token-pooling')
                descr_context = tf.reduce_max(descr_embedding, axis=1, name='descr-pooling')
                descr_neg_context = tf.reduce_max(descr_neg_embedding, axis=1, name='descr-pooling')

            # Description embedding
            self.description_embedding = descr_context

            # Code Fusion Layer
            code_concat = tf.concat([name_context, api_context, token_context],
                                    axis=1, name='code-concat')
            self.code_embedding = tf.layers.dense(inputs=code_concat,
                                                  units=embed_size,
                                                  activation=tf.nn.tanh,
                                                  name='code-fusion')

            # Normalize embeddings to prepare for cosine similarity
            normalized_code = tf.math.l2_normalize(self.code_embedding, axis=1)
            normalied_descr = tf.math.l2_normalize(descr_context, axis=1)
            normalized_descr_neg = tf.math.l2_normalize(descr_neg_context, axis=1)

            self.loss_op = tf.reduce_sum(
                tf.nn.relu(
                    tf.constant(self.params.margin, dtype=tf.float32) -
                    tf.reduce_sum(normalied_descr * normalized_code, axis=1) +
                    tf.reduce_sum(normalized_descr_neg * normalized_code, axis=1)
                )
            )

    def _make_training_step(self):
        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.loss_op, trainable_vars)
        clipped_grad, _ = tf.clip_by_global_norm(gradients, self.params.gradient_clip)
        pruned_gradients = []
        for grad, var in zip(clipped_grad, trainable_vars):
            if grad is not None:
                pruned_gradients.append((grad, var))

        self.optimizer_op = self.optimizer.apply_gradients(pruned_gradients)

    def _rnn_embedding(self, placeholder, len_placeholder, name):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.params.rnn_units, activation=tf.nn.tanh,
                                          name=name + '-fw')
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.params.rnn_units, activation=tf.nn.tanh,
                                          name=name + '-bw')

        initial_state_fw = cell_fw.zero_state(tf.shape(placeholder)[0], dtype=tf.float32)
        initial_state_bw = cell_bw.zero_state(tf.shape(placeholder)[0], dtype=tf.float32)
        emb, state = tf.nn.bidirectional_dynamic_rnn(
                                        cell_fw=cell_fw,
                                        cell_bw=cell_bw,
                                        inputs=placeholder,
                                        initial_state_fw=initial_state_fw,
                                        initial_state_bw=initial_state_bw,
                                        dtype=tf.float32,
                                        scope=name)
        return emb, state

    def _conv_1d_embedding(self, placeholder, len_placeholder, name):
        embedding = tf.layers.conv1d(inputs=placeholder,
                                     filters=self.params.embedding_size,
                                     kernel_size=self.params.kernel_size,
                                     padding='same',
                                     activation=tf.nn.tanh,
                                     name=name + '-conv-emb')

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
                                  name=name + '-attn-weights')
        alphas = tf.nn.softmax(weights, name=name + '-attn')
        return tf.reduce_sum(alphas * inputs, axis=1, name=name + '-attn-reduce')

    def _reduction_layer(self, rnn_embedding, output_size, len_placeholder, name):
        concat_tensor = tf.concat([rnn_embedding[0], rnn_embedding[1]], axis=2,
                                  name=name + '-concat')
        reduction = tf.layers.dense(inputs=concat_tensor,
                                    units=output_size,
                                    use_bias=False,
                                    name=name + '-dense')

        mask = self._create_mask(reduction, len_placeholder)

        return reduction + mask

    def _create_mask(self, placeholder, len_placeholder):
        # We mask out elements which are padded before feeding tokens into an aggregation layer
        index_list = tf.range(self.params.max_seq_length)  # S
        index_tensor = tf.tile(tf.expand_dims(index_list, axis=0),
                               multiples=(tf.shape(placeholder)[0], 1))  # B x S

        mask = index_tensor < tf.expand_dims(len_placeholder, axis=1)  # B x S

        mask = tf.tile(tf.expand_dims(mask, axis=2),  # B x S x E
                       multiples=(1, 1, self.params.embedding_size))

        return (1 - tf.cast(mask, dtype=tf.float32)) * -BIG_NUMBER

    def _generate_neg_javadoc(self, javadoc, javadoc_len):
        neg_javadoc = []
        neg_javadoc_len = []

        for i in range(len(javadoc)):
            rand_index = np.random.randint(0, len(javadoc))
            while lst_equal(javadoc[i], javadoc[rand_index]):
                rand_index = np.random.randint(0, len(javadoc))
            neg_javadoc.append(javadoc[rand_index])
            neg_javadoc_len.append(javadoc_len[rand_index])

        assert len(neg_javadoc) == len(javadoc)

        return neg_javadoc, neg_javadoc_len
