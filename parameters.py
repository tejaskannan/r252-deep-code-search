import json


class Parameters:
    """Wrapper class for model hyperparameters."""

    def __init__(self, step_size, gradient_clip,
                 max_vocab_size, max_seq_length, margin, rnn_units, hidden_dense_units,
                 hidden_fusion_units, embedding_size, batch_size, num_epochs, optimizer,
                 combine_type, seq_embedding, kernel_size, loss_func):
        self.step_size = step_size
        self.gradient_clip = gradient_clip
        self.margin = margin
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.rnn_units = rnn_units
        self.hidden_dense_units = hidden_dense_units
        self.hidden_fusion_units = hidden_fusion_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.embedding_size = embedding_size
        self.combine_type = combine_type
        self.seq_embedding = seq_embedding
        self.kernel_size = kernel_size
        self.loss_func = loss_func

    def as_dict(self):
        """Returns the parameters object as a dictionary."""
        return {
            'step_size': self.step_size,
            'gradient_clip': self.gradient_clip,
            'margin': self.margin,
            'max_vocab_size': self.max_vocab_size,
            'max_seq_length': self.max_seq_length,
            'rnn_units': self.rnn_units,
            'hidden_dense_units': self.hidden_dense_units,
            'hidden_fusion_units': self.hidden_fusion_units,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'optimizer': self.optimizer,
            'embedding_size': self.embedding_size,
            'combine_type': self.combine_type,
            'seq_embedding': self.seq_embedding,
            'kernel_size': self.kernel_size,
            'loss_func': self.loss_func
        }

    def __str__(self):
        return str(self.as_dict())


def params_dict_from_json(params_file_path, default):
    """
    Parses the given parameters file into a dictionary. Uses the default
    values when no values exist in the given file.
    """
    params_dict = default.copy()

    with open(params_file_path, 'r') as params_file:
        params_json = json.loads(params_file.read())

        for field_name, field_value in params_json.items():
            field_name = field_name.strip()
            if not (field_name in params_dict):
                print('Unrecognized parameter: {0}'.format(field_name))
                continue

            params_dict[field_name] = field_value

    return params_dict


def params_from_dict(params_dict):
    """Returns a parameters object created from the given dictionary."""
    return Parameters(
            step_size=params_dict['step_size'],
            gradient_clip=params_dict['gradient_clip'],
            margin=params_dict['margin'],
            max_vocab_size=params_dict['max_vocab_size'],
            max_seq_length=params_dict['max_seq_length'],
            rnn_units=params_dict['rnn_units'],
            hidden_dense_units=params_dict['hidden_dense_units'],
            hidden_fusion_units=params_dict['hidden_fusion_units'],
            embedding_size=params_dict['embedding_size'],
            batch_size=params_dict['batch_size'],
            num_epochs=params_dict['num_epochs'],
            optimizer=params_dict['optimizer'],
            combine_type=params_dict['combine_type'],
            seq_embedding=params_dict['seq_embedding'],
            kernel_size=params_dict['kernel_size'],
            loss_func=params_dict['loss_func']
        )
