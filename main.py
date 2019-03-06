import sys
import os
import argparse
import time
import numpy as np

from parser import Parser
from model import Model
from parameters import params_from_dict, params_dict_from_json
from search import DeepCodeSearchDB
from utils import value_if_non_empty, write_methods_to_file
from utils import load_parameters, add_slash_to_end
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

def parse_args():
    arg_parser = argparse.ArgumentParser(description='Deep Code Search')

    # Action arguments
    arg_parser.add_argument('--generate', action='store_true', help='Generate a dataset.')
    arg_parser.add_argument('--train', action='store_true', help='Train a model.')
    arg_parser.add_argument('--index', action='store_true', help='Index a corpus.')
    arg_parser.add_argument('--search', action='store_true', help='Execute a search query.')
    arg_parser.add_argument('--overlap', action='store_true', help='Compute vocabulary overlap.')

    # Input Arguments
    arg_parser.add_argument('--input', type=str, default='', help='Input file, folder, or string.')
    arg_parser.add_argument('--output', type=str, default='', help='Output file or folder.')
    arg_parser.add_argument('--model', type=str, default='', help='Directory of saved model.')
    arg_parser.add_argument('--params', type=str, default='', help='Parameters file.')
    arg_parser.add_argument('--table', type=str, default='code', help='Table name to store an indexed corpus.')
    arg_parser.add_argument('--threshold', type=int, default=2, help='Threshold value for methods or search results.')
    arg_parser.add_argument('--train-dir', type=str, default='train_data/', help='Directory with training data.')
    arg_parser.add_argument('--valid-dir', type=str, default='validation_data/', help='Direction with validation data.')
    arg_parser.add_argument('--hit-rank', action='store_true', help='Calculate hit ranks after indexing.')
    arg_parser.add_argument('--subtokenize', action='store_true', help='Subtokenize API calls during generation.')

    return arg_parser.parse_args()

def main():
    args = parse_args()
    print(args)
    
    # Restore parameters from the given checkpoint
    params = default_params
    if len(args.model) > 0:
        params = load_parameters(args.model)
    elif len(args.params) > 0:
        params = params_dict_from_json(args.params, params)
    params = params_from_dict(params)

    if args.generate:
        if len(args.input) == 0:
            print('Must specify an input folder or file.')
            return

        args.output = value_if_non_empty(args.output, 'data/')
        args.output = add_slash_to_end(args.output)

        parser = Parser('filters/tags.txt', 'filters/stopwords.txt', args.threshold)
        if args.input[-1] == '/':
            written = parser.generate_data_from_dir(args.input, args.output,
                                                    should_subtokenize=args.subtokenize)
        else:
            written = parser.generate_data_from_file(args.input, args.output,
                                                     should_subtokenize=args.subtokenize)
    elif args.train:
        args.output = value_if_non_empty(args.output, 'trained_models/')
        model = Model(params, args.train_dir, args.valid_dir, args.output)
        model.train()
    elif args.index:
        if len(args.input) == 0:
            print('Must specify a corpus to index.')
            return

        if len(args.model) == 0:
            print('Must specify a model to use.')
            return
        args.model = add_slash_to_end(args.model)

        args.output = value_if_non_empty(args.output, 'trained_models/')
        args.output = add_slash_to_end(args.output)

        model = Model(params, args.train_dir, args.valid_dir, args.output)
        model.restore(args.model)

        db = DeepCodeSearchDB(table=args.table, model=model,
                              embedding_size=params.embedding_size)

        written = 0
        if args.input[-1] == '/':
            written = db.index_dir(args.input)
        else:
            written = db.index_file(args.input)
        print('Indexed {0} methods into table: {1}'.format(written, args.table))

        # We only support hit rank calculations when indexing a directory
        if args.hit_rank and args.input[-1] == '/':
            print('Calculating Hit Ranks')
            success_hit_rank, mean_hit_rank = db.hit_rank_over_corpus(args.input)
            print('Success Hit Rank: {0}'.format(success_hit_rank))
            print('Mean Hit Rank: {0}'.format(mean_hit_rank))
    elif args.search:
        if len(args.input) == 0:
            print('Must specify a query or query file.')
            return

        if len(args.model) == 0:
            print('Must specify a model to use.')
            return
        args.model = add_slash_to_end(args.model)

        args.output = value_if_non_empty(args.output, 'searches/')
        args.output = add_slash_to_end(args.output)

        model = Model(params, args.train_dir, args.valid_dir, args.output)
        model.restore(args.model)

        db = DeepCodeSearchDB(table=args.table, model=model,
                              embedding_size=params.embedding_size)

        search_queries = []
        if '.txt' in args.input:
            with open(args.input, 'r') as query_file:
                for query in query_file:
                    search_queries.append(query.strip())
        else:
            search_queries.append(args.input)

        args.output = value_if_non_empty(args.output, 'searches/')
        args.output = add_slash_to_end(args.output)
        out_folder = add_slash_to_end(args.output + args.model.split('/')[1])
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        print(out_folder)

        times = []
        for query in search_queries:
            start = time.time()
            results = db.search(query, k=args.threshold)
            end = time.time()
            times.append(end - start)

            results = list(map(lambda r: str(r.decode('utf-8')), results))

            output_file = out_folder + '/' + query.replace(' ', '_') + '.txt'
            write_methods_to_file(output_file, results)

        print('Average Query Time: {0}s'.format(np.average(times)))
    elif args.overlap:
        if len(args.model) == 0:
            print('Must specify a model to use.')
            return
        if len(args.input) == 0:
            print('Must specify an input folder.')
            return

        args.input = add_slash_to_end(args.input)
        args.model = add_slash_to_end(args.model)
        args.output = value_if_non_empty(args.output, 'trained_models/')
        args.output = add_slash_to_end(args.output)

        model = Model(params, args.train_dir, args.valid_dir, args.output)
        model.restore(args.model)

        db = DeepCodeSearchDB(table=args.table, model=model,
                              embedding_size=params.embedding_size)
        overlaps, totals = db.vocabulary_overlap(args.input)
        labels = ['Method Tokens', 'Method API Calls', 'Method Names', 'Total']

        for i in range(len(overlaps)):
            frac = overlaps[i] / totals[i]
            print(OVERLAP_FORMAT.format(labels[i], overlaps[i], totals[i], round(frac, 4)))


if __name__ == '__main__':
    main()
