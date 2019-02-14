import sys
import getopt
from parser import Parser
from model import Model


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"gtxi:o:f:v:l:r:",["generate","train","index", "input", "output", "train-dir", "valid-dir", "log-dir", "restore"])
    except getopt.GetoptError:
      print("Incorrect Arguments.")
      sys.exit(0)

    inpt = ""
    outpt = ""
    train_dir = ""
    valid_dir = ""
    log_dir = ""
    restore_dir = ""
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

    for opt, arg in opts:
        if opt in ("-g", "--generate"):
            parser = Parser("filters/tags.txt", "filters/stopwords.txt")
            if len(inpt) == 0:
                print("Must specify an input folder or file.")
                sys.exit(0)

            parser = Parser("filters/tags.txt", "filters/stopwords.txt")
            out_folder = outpt if len(outpt) > 0 else "data/"
            if inpt[-1] == "/":
                written = parser.parse_directory(inpt, out_folder)
            else:
                written = parser.parse_file(inpt, out_folder)
            print("Generated dataset size: {0}".format(written))
        if opt in ("-t", "--train"):
            train_dir = train_dir if len(train_dir) > 0 else "train_data/"
            valid_dir = valid_dir if len(valid_dir) > 0 else "validation_data/"
            save_dir = outpt if len(outpt) > 0 else "trained_models/"
            log_dir = log_dir if len(log_dir) > 0 else "log/"
            model = Model(train_dir, valid_dir, save_dir, log_dir)
            model.train()
        if opt in ("-x", "--index"):
            if len(restore_dir) == 0:
                print("Must specify a model to use.")
                sys.exit(0)
            if restore_dir[-1] != "/":
                restore_dir += "/"

            train_dir = train_dir if len(train_dir) > 0 else "train_data/"
            valid_dir = valid_dir if len(valid_dir) > 0 else "validation_data/"
            save_dir = outpt if len(outpt) > 0 else "trained_models/"
            log_dir = log_dir if len(log_dir) > 0 else "log/"
            model = Model(train_dir, valid_dir, save_dir, log_dir)
            model.restore(restore_dir)

if __name__ == '__main__':
    main(sys.argv[1:])