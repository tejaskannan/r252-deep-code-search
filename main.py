import sys
import os
import getopt
import time
import numpy as np

from parser import Parser
from model import Model
from parameters import params_from_dict, params_dict_from_json
from search import DeepCodeSearchDB
from utils import try_parse_int, value_if_non_empty, write_methods_to_file, add_slash_to_end
from utils import load_parameters
from constants import OVERLAP_FORMAT

default_params = {
    'step_size': 0.001,
    'gradient_clip': 1,
    'margin': 0.05,
    'max_vocab_size': 10000,
    'max_seq_length': 100,
    'rnn_units': 64,
    'hidden_dense_units': 64,
    'hidden_fusion_units': 128,
    'embedding_size': 64,
    'batch_size': 32,
    'num_epochs': 4,
    'optimizer': 'adam',
    'combine_type': 'max_pool',
    'seq_embedding': 'RNN',
    'kernel_size': 5
}


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'gtxrhi:o:t:v:l:m:p:n:s:k:', ['generate', 'train', 'index', 'input', 'output', 'train-dir', 'valid-dir', 'log-dir', 'model', 'params', 'table-name', 'search', 'overlap', 'hit-rank'])
    except getopt.GetoptError:
        print('Incorrect Arguments.')
        sys.exit(0)

    params = default_params
    inpt = ''
    outpt = ''
    train_dir = ''
    valid_dir = ''
    log_dir = ''
    restore_dir = ''
    table_name = ''
    search_query = ''
    threshold = ''
    should_calc_hit_rank = False
    for opt, arg in opts:
        if opt in ('-i', '--input'):
            inpt = arg
        if opt in ('-o', '--output'):
            outpt = arg
        if opt in ('-t', '--train-dir'):
            train_dir = arg
        if opt in ('-v', '--valid-dir'):
            valid_dir = arg
        if opt in ('-l', '--log-dir'):
            log_dir = arg
        if opt in ('-m', '--model'):
            restore_dir = arg
        if opt in ('-p', '--params'):
            params_file = arg
            params = params_dict_from_json(params_file, params)
        if opt in ('-n', '--table-name'):
            table_name = arg
        if opt in ('-s', '--search'):
            search_query = arg
        if opt == '-k':
            threshold = arg
        if opt in ('-h', '--hit-rank'):
            should_calc_hit_rank = True

    # Restore parameters from the given checkpoint
    if len(restore_dir) > 0:
        params = load_parameters(restore_dir)

    params = params_from_dict(params)
    for opt, arg in opts:
        if opt in ('-g', '--generate'):
            if len(inpt) == 0:
                print('Must specify an input folder or file.')
                sys.exit(0)

            threshold = try_parse_int(threshold, 3)
            parser = Parser('filters/tags.txt', 'filters/stopwords.txt', threshold)
            out_folder = outpt if len(outpt) > 0 else 'data/'
            if inpt[-1] == '/':
                written = parser.generate_data_from_dir(inpt, out_folder)
            else:
                written = parser.generate_data_from_file(inpt, out_folder)
            print('Generated dataset size: {0}'.format(written))
        if opt in ('-t', '--train'):
            train_dir = value_if_non_empty(train_dir, 'train_data/')
            valid_dir = value_if_non_empty(valid_dir, 'validation_data/')
            save_dir = value_if_non_empty(outpt, 'trained_models/')
            log_dir = value_if_non_empty(log_dir, 'log/')

            model = Model(params, train_dir, valid_dir, save_dir, log_dir)
            model.train()
        if opt in ('-x', '--index'):
            if len(restore_dir) == 0:
                print('Must specify a model to use.')
                sys.exit(0)
            if restore_dir[-1] != '/':
                restore_dir += '/'

            if len(inpt) == 0:
                print('Must specify a file to index.')
                sys.exit(0)

            train_dir = value_if_non_empty(train_dir, 'train_data/')
            valid_dir = value_if_non_empty(valid_dir, 'validation_data/')
            save_dir = value_if_non_empty(outpt, 'trained_models/')
            log_dir = value_if_non_empty(log_dir, 'log/')

            model = Model(params, train_dir, valid_dir, save_dir, log_dir)
            model.restore(restore_dir)

            table_name = value_if_non_empty(table_name, 'code')
            db = DeepCodeSearchDB(table=table_name, model=model,
                                  embedding_size=params.embedding_size)

            written = 0
            if inpt[-1] == '/':
                written = db.index_dir(inpt)
            else:
                written = db.index_file(inpt)
            print('Indexed {0} methods into table: {1}'.format(written, table_name))

            # We only support hit rank calculations when indexing a directory
            if should_calc_hit_rank and inpt[-1] == '/':
                print('Calculating Hit Ranks')
                success_hit_rank, mean_hit_rank = db.hit_rank_over_corpus(inpt)
                print('Success Hit Rank: {0}'.format(success_hit_rank))
                print('Mean Hit Rank: {0}'.format(mean_hit_rank))

        if opt in ('-s', '--search'):
            if len(search_query) == 0:
                print('Must specify a query or query file.')
                sys.exit(0)

            if len(restore_dir) == 0:
                print('Must specify a model to use.')
                sys.exit(0)
            if restore_dir[-1] != '/':
                restore_dir += '/'

            train_dir = value_if_non_empty(train_dir, 'train_data/')
            valid_dir = value_if_non_empty(valid_dir, 'validation_data/')
            save_dir = value_if_non_empty(outpt, 'trained_models/')
            log_dir = value_if_non_empty(log_dir, 'log/')

            model = Model(params, train_dir, valid_dir, save_dir, log_dir)
            model.restore(restore_dir)

            table_name = value_if_non_empty(table_name, 'code')
            db = DeepCodeSearchDB(table=table_name, model=model,
                                  embedding_size=params.embedding_size)

            search_queries = []
            if '.txt' in search_query:
                with open(search_query, 'r') as query_file:
                    for query in query_file:
                        search_queries.append(query.strip())
            else:
                search_queries.append(search_query)

            threshold = try_parse_int(threshold, 10)

            out_folder = 'searches/' + restore_dir.split('/')[1]
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)

            times = []
            for query in search_queries:
                start = time.time()
                results = db.search_full(query, k=threshold)
                end = time.time()
                times.append(end - start)

                results = list(map(lambda r: str(r.decode('utf-8')), results))

                output_file = out_folder + '/' + query.replace(' ', '_') + '.txt'
                write_methods_to_file(output_file, results)

            print('Average Query Time: {0}s'.format(np.average(times)))
        if opt in ('-r', '--overlap'):
            if len(restore_dir) == 0:
                print('Must specify a model to use.')
                sys.exit(0)
            if len(inpt) == 0:
                print('Must specify an input folder.')
                sys.exit(0)

            inpt = add_slash_to_end(inpt)
            restore_dir = add_slash_to_end(restore_dir)

            train_dir = value_if_non_empty(train_dir, 'train_data/')
            valid_dir = value_if_non_empty(valid_dir, 'validation_data/')
            save_dir = value_if_non_empty(outpt, 'trained_models/')
            log_dir = value_if_non_empty(log_dir, 'log/')

            model = Model(params, train_dir, valid_dir, save_dir, log_dir)
            model.restore(restore_dir)

            table_name = value_if_non_empty(table_name, 'code')
            db = DeepCodeSearchDB(table=table_name, model=model,
                                  embedding_size=params.embedding_size)
            overlaps, totals = db.vocabulary_overlap(inpt)
            labels = ['Method Tokens', 'Method API Calls', 'Method Names', 'Total']

            for i in range(len(overlaps)):
                frac = overlaps[i] / totals[i]
                print(OVERLAP_FORMAT.format(labels[i], overlaps[i], totals[i], round(frac, 4)))


if __name__ == '__main__':
    main(sys.argv[1:])
