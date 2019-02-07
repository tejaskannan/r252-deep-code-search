
class Parameters:

    def __init__(self, train_frac, valid_frac, step_size, gradient_clip,
                 max_vocab_size, seq_length, margin, rnn_units, dense_units,
                 batch_size, num_epochs, optimizer):
        self.train_frac = train_frac
        self.valid_frac = valid_frac
        self.step_size = step_size
        self.gradient_clip = gradient_clip
        self.margin = margin
        self.max_vocab_size = max_vocab_size
        self.seq_length = seq_length
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer

    def as_dict(self):
        return {
            "train_frac": self.train_frac,
            "valid_frac": self.valid_frac,
            "step_size": self.step_size,
            "gradient_clip": self.gradient_clip,
            "margin": self.margin,
            "max_vocab_size": self.max_vocab_size,
            "seq_length": self.seq_length,
            "rnn_units": self.rnn_units,
            "dense_units": self.dense_units,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "optimizer": self.optimizer
        }
