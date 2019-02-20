import sys
import getopt
import subprocess
from parser import Parser
from model import Model
from parameters import params_from_dict, params_dict_from_json
from search import DeepCodeSearchDB
from utils import try_parse_int, value_if_non_empty, write_to_file

default_params = {
    "step_size" : 0.001,
    "gradient_clip" : 1,
    "margin" : 0.05,
    "max_vocab_size" : 10000,
    "max_seq_length" : 50,
    "rnn_units" : 64,
    "dense_units" : 64,
    "embedding_size" : 64,
    "batch_size" : 32,
    "num_epochs" : 4,
    "optimizer" : "adam",
    "combine_type": "attention"
}

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"gtxi:o:f:v:l:r:p:n:s:k:",["generate","train","index", "input", "output", "train-dir", "valid-dir", "log-dir", "restore", "params", "table-name", "search"])
    except getopt.GetoptError:
      print("Incorrect Arguments.")
      sys.exit(0)

    params = default_params
    inpt = ""
    outpt = ""
    train_dir = ""
    valid_dir = ""
    log_dir = ""
    restore_dir = ""
    table_name = ""
    search_query = ""
    threshold = ""
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inpt = arg
        if opt in ("-o", "--output"):
            outpt = arg
        if opt in ("-f", "--train-dir"):
            train_dir = arg
        if opt in ("-v", "--valid-dir"):
            valid_dir = arg
        if opt in ("-l", "--log-dir"):
            log_dir = arg
        if opt in ("-r", "--restore"):
            restore_dir = arg
        if opt in ("-p", "--params"):
            params_file = arg
            params = params_dict_from_json(params_file, params)
        if opt in ("-n", "--table-name"):
            table_name = arg
        if opt in ("-s", "--search"):
            search_query = arg
        if opt == "-k":
            threshold = arg

    params = params_from_dict(params)
    for opt, arg in opts:
        if opt in ("-g", "--generate"):
            if len(inpt) == 0:
                print("Must specify an input folder or file.")
                sys.exit(0)

            threshold = try_parse_int(threshold, 3)
            parser = Parser("filters/tags.txt", "filters/stopwords.txt", threshold)
            out_folder = outpt if len(outpt) > 0 else "data/"
            if inpt[-1] == "/":
                written = parser.generate_data_from_dir(inpt, out_folder)
            else:
                written = parser.generate_data_from_file(inpt, out_folder)
            print("Generated dataset size: {0}".format(written))
        if opt in ("-t", "--train"):
            train_dir = value_if_non_empty(train_dir, "train_data/")
            valid_dir = value_if_non_empty(valid_dir, "validation_data/")
            save_dir = value_if_non_empty(outpt, "trained_models/")
            log_dir = value_if_non_empty(log_dir, "log/")

            model = Model(params, train_dir, valid_dir, save_dir, log_dir)
            model.train()
        if opt in ("-x", "--index"):
            if len(restore_dir) == 0:
                print("Must specify a model to use.")
                sys.exit(0)
            if restore_dir[-1] != "/":
                restore_dir += "/"

            if len(inpt) == 0:
                print("Must specify a file to index.")
                sys.exit(0)

            train_dir = value_if_non_empty(train_dir, "train_data/")
            valid_dir = value_if_non_empty(valid_dir, "validation_data/")
            save_dir = value_if_non_empty(outpt, "trained_models/")
            log_dir = value_if_non_empty(log_dir, "log/")

            model = Model(params, train_dir, valid_dir, save_dir, log_dir)
            model.restore(restore_dir)

            table_name = value_if_non_empty(table_name, "code")
            db = DeepCodeSearchDB(table=table_name, model=model,
                                  embedding_size=params.embedding_size)

            written = 0
            if inpt[-1] == "/":
                written = db.index_dir(inpt)
            else:
                written = db.index_file(inpt)
            print("Indexed {0} methods into table: {1}".format(written, table_name))
        if opt in ("-s", "--search"):
            if len(search_query) == 0:
                print("Must specify a query.")
                sys.exit(0)

            if len(restore_dir) == 0:
                print("Must specify a model to use.")
                sys.exit(0)
            if restore_dir[-1] != "/":
                restore_dir += "/"

            if len(outpt) == 0:
                print("Must specify an output file to use.")
                sys.exit(0)

            train_dir = value_if_non_empty(train_dir, "train_data/")
            valid_dir = value_if_non_empty(valid_dir, "validation_data/")
            save_dir = value_if_non_empty(outpt, "trained_models/")
            log_dir = value_if_non_empty(log_dir, "log/")

            model = Model(params, train_dir, valid_dir, save_dir, log_dir)
            model.restore(restore_dir)

            table_name = value_if_non_empty(table_name, "code")
            db = DeepCodeSearchDB(table=table_name, model=model,
                                  embedding_size=params.embedding_size)

            threshold = try_parse_int(threshold, 10)
            results = db.search(search_query, k=threshold)
            results = list(map(lambda r: str(r.decode("utf-8")), results))
            write_to_file(outpt, results)
            

if __name__ == '__main__':
    main(sys.argv[1:])