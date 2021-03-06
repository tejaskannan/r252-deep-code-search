import numpy as np
import tensorflow as tf
import gzip
import pickle
from os import mkdir
from os.path import exists
from datetime import datetime
from dpu_utils.mlutils import Vocabulary
from parameters import Parameters, params_from_dict
from dataset import Dataset, Batch
from utils import lst_equal, log_record, add_slash_to_end
from constants import *


class Model:
    """Class which implements the Deep Code Search model."""

    def __init__(self, params, train_dir, valid_dir, save_dir):
        self.params = params

        self.train_dir = add_slash_to_end(train_dir)
        self.valid_dir = add_slash_to_end(valid_dir)
        self.save_dir = add_slash_to_end(save_dir)

        self.scope = 'deep-cs'

        self.dataset = Dataset(train_dir=train_dir,
                               valid_dir=valid_dir,
                               max_seq_length=self.params.max_seq_length,
                               max_vocab_size=self.params.max_vocab_size)

        self._sess = tf.Session(graph=tf.Graph())

        max_seq_len = self.params.max_seq_length
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
        """Trains a model using the class's parameters."""

        with self._sess.graph.as_default():

            init_op = tf.variables_initializer(self._sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self._sess.run(init_op)

            # Initializes a name for this training run using the current time
            train_time = datetime.now().strftime(DATE_FORMAT)
            train_name = NAME_FORMAT.format(self.params.combine_type, self.params.seq_embedding, train_time)
            overfit_train_name = 'overfit-' + train_name

            output_dir = self.save_dir + train_name
            if not exists(output_dir):
                mkdir(output_dir)

            # Initalize logging of this training run
            csv_name = LOG_FORMAT.format(self.save_dir, train_name)
            log_record(csv_name, ['Epoch', 'Avg Train Loss', 'Avg Validation Loss'])

            # Initialize best validation loss to a large value
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

                # Execute training batches
                for i in range(num_train_batches):
                    feed_dict = self._create_feed_dict_from_batch(train_batches, i)
                    ops = [self.loss_op, self.optimizer_op]
                    op_result = self._sess.run(ops, feed_dict=feed_dict)

                    avg_train_loss = (op_result[0]) / self.params.batch_size
                    train_losses.append(avg_train_loss)

                    print('Training batch {0}/{1}: {2}'.format(i, num_train_batches-1, avg_train_loss))

                print(LINE)

                # Execute validation batches
                for i in range(num_valid_batches):
                    feed_dict = self._create_feed_dict_from_batch(valid_batches, i)
                    valid_result = self._sess.run(self.loss_op, feed_dict=feed_dict)
                    avg_valid_loss = valid_result / self.params.batch_size
                    valid_losses.append(avg_valid_loss)

                    print('Validation batch {0}/{1}: {2}'.format(i, num_valid_batches-1, avg_valid_loss))

                avg_valid_loss = np.average(valid_losses)
                avg_train_loss = np.average(train_losses)

                # Log average training and validation losses for this epoch.
                log_record(csv_name, [str(epoch), str(avg_train_loss), str(avg_valid_loss)])

                print(LINE)
                print('Average training loss in epoch {0}: {1}'.format(epoch, avg_train_loss))
                print('Average validation loss in epoch {0}: {1}'.format(epoch, avg_valid_loss))

                # Save model if it displays the best validation loss.
                if (avg_valid_loss < best_valid_loss):
                    best_valid_loss = avg_valid_loss
                    print('Saving model: ' + train_name)
                    self.save(self.save_dir, train_name)

                print(LINE)

    def save(self, base_folder, name):
        """Saves this model and all associated parameters to the given folder"""

        meta_data = {
            'model_type': type(self).__name__,
            'parameters': self.params.as_dict(),
            'name': name,
            'vocabulary': self.dataset.vocabulary
        }

        save_folder = base_folder + name
        if not exists(save_folder):
            mkdir(save_folder)

        # Save metadata
        meta_path = save_folder + '/' + META_NAME
        with gzip.GzipFile(meta_path, 'wb') as out_file:
            pickle.dump(meta_data, out_file)

        # Save Tensorflow model
        model_path = save_folder + '/' + MODEL_NAME
        saver = tf.train.Saver()
        saver.save(self._sess, model_path)

    def restore(self, save_folder):
        """
        Restores the model saved in the given folder. The parameter
        string is assumed to end in a slash.
        """

        # Restore metadata
        meta_data = {}
        meta_path = save_folder + META_NAME
        with gzip.GzipFile(meta_path, 'rb') as in_file:
            meta_data = pickle.load(in_file)

        self.params = params_from_dict(meta_data['parameters'])
        self.dataset.vocabulary = meta_data['vocabulary']

        # Restore Tensorflow model parameters
        with self._sess.graph.as_default():
            model_path = save_folder + MODEL_NAME
            saver = tf.train.Saver()
            saver.restore(self._sess, model_path)

    def embed_method(self, method_name, method_api, method_tokens):
        """
        Returns an embedding vector for a method with the given name,
        API calls, and method tokens.
        """

        # Tensorize inputs
        name_vec, name_len = self.dataset.create_tensor(method_name)
        api_vec, api_len = self.dataset.create_tensor(method_api)
        token_vec, token_len = self.dataset.create_tensor(method_tokens)

        # Reshape the inputs to match the dimensions expected by the Tensorflow model
        name_tensor = np.array([name_vec])
        name_len_tensor = np.array([name_len])
        api_tensor = np.array([api_vec])
        api_len_tensor = np.array([api_len])
        token_tensor = np.array([token_vec])
        token_len_tensor = np.array([token_len])

        with self._sess.graph.as_default():

            feed_dict = {
                self.method_names: name_tensor,
                self.method_names_len: name_len_tensor,
                self.method_apis: api_tensor,
                self.method_apis_len: api_len_tensor,
                self.method_tokens: token_tensor,
                self.method_tokens_len: token_len_tensor,
            }

            embedding = self._sess.run(self.code_embedding, feed_dict=feed_dict)

        return embedding[0]

    def embed_description(self, description):
        """Returns the embedding for the given description."""

        descr_vec, descr_len = self.dataset.create_tensor(description)

        with self._sess.graph.as_default():
            feed_dict = {
                self.description: np.array([descr_vec]),
                self.description_len: np.array([descr_len])
            }

            embedding = self._sess.run(self.description_embedding, feed_dict=feed_dict)

        return embedding[0]

    def _make_model(self):
        """
        Creates the computational graph which implementes the Deep Code Search model.
        The created model variation is dictated by the parameters provided to this class.

        """

        vocab_size = self.params.max_vocab_size
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

            # Embed sequences. By default we use a BiRNN.
            if self.params.seq_embedding == 'conv':
                name_embedding = self._conv_1d_embedding(name_encoding, name='name-embed')
                api_embedding = self._conv_1d_embedding(api_encoding, name='api-embed')
                descr_embedding = self._conv_1d_embedding(descr_encoding, name='descr-embed')
                descr_neg_embedding = self._conv_1d_embedding(descr_neg_encoding, name='descr-embed')
            else:
                # Embeddings using a BiRNN
                name_emb, _name_state = self._rnn_embedding(name_encoding, name='name-rnn')
                api_emb, _api_state = self._rnn_embedding(api_encoding, name='api-rnn')
                descr_emb, _descr_state = self._rnn_embedding(descr_encoding, name='descr-rnn')
                descr_neg_emb, _descr_neg_state = self._rnn_embedding(descr_neg_encoding, name='descr-rnn')

                # Combine the outputs from the forward and backward passes
                name_embedding = self._reduction_layer(name_emb, embed_size, name='name-embed')
                api_embedding = self._reduction_layer(api_emb, embed_size, name='api-embed')
                descr_embedding = self._reduction_layer(descr_emb, embed_size, name='descr-embed')
                descr_neg_embedding = self._reduction_layer(descr_neg_emb, embed_size, name='descr-embed')

            # Create masks based on sequence length
            name_mask = self._create_mask(name_embedding, self.method_names_len)
            api_mask = self._create_mask(api_embedding, self.method_apis_len)
            token_mask = self._create_mask(token_embedding, self.method_tokens_len)
            descr_mask = self._create_mask(descr_embedding, self.description_len)
            descr_neg_mask = self._create_mask(descr_neg_embedding, self.description_neg_len)

            # Combine sequences into a single context vector. By default we use max pooling.
            if self.params.combine_type == 'attention':
                name_context = self._attention_layer(name_embedding, name_mask, name='name-attn')
                api_context = self._attention_layer(api_embedding, api_mask, name='api-attn')
                token_context = self._attention_layer(token_embedding, token_mask, name='token-attn')
                descr_context = self._attention_layer(descr_embedding, descr_mask, name='descr-attn')
                descr_neg_context = self._attention_layer(descr_neg_embedding, descr_neg_mask, name='descr-attn')
            else:
                # Max Pooling
                name_context = tf.reduce_max(name_embedding + name_mask, axis=1, name='name-pooling')
                api_context = tf.reduce_max(api_embedding + api_mask, axis=1, name='api-pooling')
                token_context = tf.reduce_max(token_embedding + token_mask, axis=1, name='token-pooling')
                descr_context = tf.reduce_max(descr_embedding + descr_mask, axis=1, name='descr-pooling')
                descr_neg_context = tf.reduce_max(descr_neg_embedding + descr_neg_mask, axis=1, name='descr-pooling')

            # Description embeddings
            self.description_embedding = descr_context
            self.neg_descr_embedding = descr_neg_context

            # Code Fusion Layer
            code_concat = tf.concat([name_context, api_context, token_context],
                                    axis=1, name='code-concat')
            self.code_embedding = tf.layers.dense(inputs=code_concat,
                                                  units=embed_size,
                                                  activation=tf.nn.tanh,
                                                  name='code-fusion')

            # Normalize embeddings to prepare for loss calculation
            normalized_code = tf.math.l2_normalize(self.code_embedding, axis=1)
            normalized_descr = tf.math.l2_normalize(descr_context, axis=1)
            normalized_descr_neg = tf.math.l2_normalize(descr_neg_context, axis=1)

            # Loss Function. By default we use margin-based cosine simlarity.
            if self.params.loss_func == 'neg_sampling':
                sim_mat = tf.matmul(normalized_code, normalized_descr, transpose_b=True)

                neg_identity = -1.0 * tf.eye(tf.shape(sim_mat)[0], dtype=tf.float32)
                neg_sample_mask = 1.0 + neg_identity

                neg_scores = tf.reduce_sum(sim_mat * neg_sample_mask, axis=1)
                num_neg_samples = tf.cast(tf.shape(sim_mat)[1], dtype=tf.float32) - 1
                neg_scores = neg_scores / (num_neg_samples + SMALL_NUMBER)

                pos_scores = tf.reduce_sum(sim_mat * neg_identity, axis=1)

                scores = pos_scores + neg_scores
                self.loss_op = tf.reduce_sum(scores)
            else:
                self.loss_op = tf.reduce_sum(
                    tf.nn.relu(
                        tf.constant(self.params.margin, dtype=tf.float32) -
                        tf.reduce_sum(normalized_descr * normalized_code, axis=1) +
                        tf.reduce_sum(normalized_descr_neg * normalized_code, axis=1)
                    )
                )

    def _make_training_step(self):
        """Implements a step of gradient descent to optimize the model's weights."""

        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.loss_op, trainable_vars)
        clipped_grad, _ = tf.clip_by_global_norm(gradients, self.params.gradient_clip)
        pruned_gradients = []
        for grad, var in zip(clipped_grad, trainable_vars):
            if grad is not None:
                pruned_gradients.append((grad, var))

        self.optimizer_op = self.optimizer.apply_gradients(pruned_gradients)

    def _rnn_embedding(self, placeholder, name):
        """
        Implements the BiRNN token embedding layer.
        The outputs are not clipped based on sequence length.
        """

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

    def _conv_1d_embedding(self, placeholder, name):
        """
        Implements the 1D convolution used for token embeddings.
        The outputs are not clipped based on sequence length.
        """

        embedding = tf.layers.conv1d(inputs=placeholder,
                                     filters=self.params.embedding_size,
                                     kernel_size=self.params.kernel_size,
                                     padding='same',
                                     activation=tf.nn.tanh,
                                     name=name + '-conv-emb')
        return embedding

    def _attention_layer(self, inputs, input_mask, name):
        """Implements the self-attention layer."""
        weights = tf.layers.dense(inputs=inputs, units=1, activation=tf.nn.tanh,
                                  name=name + '-attn-weights')
        alphas = tf.nn.softmax(weights + input_mask, axis=1, name=name + '-attn')
        return tf.reduce_sum(inputs * alphas, axis=1, name=name + '-attn-reduce')

    def _reduction_layer(self, rnn_embedding, output_size, name):
        """
        Implements a single dense layer which is used to combine the forward and
        backward outputs from a BiRNN.
        """
        concat_tensor = tf.concat([rnn_embedding[0], rnn_embedding[1]], axis=2,
                                  name=name + '-concat')
        reduction = tf.layers.dense(inputs=concat_tensor,
                                    units=output_size,
                                    use_bias=False,
                                    name=name + '-dense')
        return reduction

    def _create_mask(self, placeholder, len_placeholder):
        """
        Returns a mask which can be used to adjust for the length
        of input token sequences. An element of this mask is 0 for indices
        less than the sequence lenth and -BIG_NUMBER for indices beyond the sequence
        length.
        """
        index_list = tf.range(self.params.max_seq_length)
        index_tensor = tf.tile(tf.expand_dims(index_list, axis=0),
                               multiples=(tf.shape(placeholder)[0], 1))

        mask = index_tensor < tf.expand_dims(len_placeholder, axis=1)
        mask = tf.expand_dims(mask, axis=2)

        return (1 - tf.cast(mask, dtype=tf.float32)) * -BIG_NUMBER

    def _generate_neg_javadoc(self, javadoc, javadoc_len):
        """
        Returns arrays of the same dimension as javadoc and javadoc_len which
        contain negative javadoc examples used during training.
        """
        neg_javadoc = []
        neg_javadoc_len = []
        neg_javadoc_freq = []

        for i in range(len(javadoc)):
            rand_index = np.random.randint(0, len(javadoc))

            # We compare the randomly selected example to ensure
            # that it is indeed a negative example. This element-by-element comparison
            # is carried out because there are examples of different methods
            # which have the same Javadoc comments.
            while lst_equal(javadoc[i], javadoc[rand_index]):
                rand_index = np.random.randint(0, len(javadoc))

            neg_javadoc.append(javadoc[rand_index])
            neg_javadoc_len.append(javadoc_len[rand_index])

        assert len(neg_javadoc) == len(javadoc)

        return neg_javadoc, neg_javadoc_len

    def _create_feed_dict_from_batch(self, batches, i):
        """Returns the feed dictionary for the given batch."""
        jd_neg_batch, jd_neg_len_batch = \
            self._generate_neg_javadoc(batches.javadoc_batches[i],
                                       batches.javadoc_len_batches[i])
        return {
            self.method_names: batches.name_batches[i],
            self.method_apis: batches.api_batches[i],
            self.method_tokens: batches.token_batches[i],
            self.description: batches.javadoc_batches[i],
            self.description_neg: jd_neg_batch,
            self.method_names_len: batches.name_len_batches[i],
            self.method_apis_len: batches.api_len_batches[i],
            self.method_tokens_len: batches.token_len_batches[i],
            self.description_len: batches.javadoc_len_batches[i],
            self.description_neg_len: jd_neg_len_batch
        }
