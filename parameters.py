
class Parameters:

    def __init__(self, step_size, gradient_clip,
                 max_vocab_size, max_seq_length, margin, rnn_units, dense_units,
                 embedding_size, batch_size, num_epochs, optimizer):
        self.step_size = step_size
        self.gradient_clip = gradient_clip
        self.margin = margin
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.embedding_size = embedding_size

    def as_dict(self):
        return {
            "step_size": self.step_size,
            "gradient_clip": self.gradient_clip,
            "margin": self.margin,
            "max_vocab_size": self.max_vocab_size,
            "max_seq_length": self.max_seq_length,
            "rnn_units": self.rnn_units,
            "dense_units": self.dense_units,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "optimizer": self.optimizer,
            "embedding_size": self.embedding_size
        }
