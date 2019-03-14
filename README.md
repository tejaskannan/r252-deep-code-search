# R252: Deep Code Search

This respository holds a Tensorflow implementation of Gu et al.'s [Deep Code Search](https://guxd.github.io/papers/deepcs.pdf). This entire project is implemented in Python 3. This implementation was completed for Cambridge's ACS module R252: Machine Learning for Programming.

The project is driven by ```main.py``` and has four major components: generating new datasets, training models, indexing data into a Redis database, and executing searches. The command-line interface for each component is described below.

The project uses [Redis](https://pypi.org/project/redis/), Spotify's [Annoy](https://github.com/spotify/annoy), Microsoft's [DPU Utilities](https://github.com/Microsoft/dpu-utils), and [Tensorflow](https://www.tensorflow.org/).

# Generating Datasets
A new dataset can be generated using the command below.
```sh
python main.py --generate --input <input-dir> --output <output-dir>
```
The input directory soecifies the source code repositories which will be used to generate the new dataset. This routine parses Java Abstract Syntax Trees (AST) which are serialized as [protocol buffers](https://developers.google.com/protocol-buffers/). This command creates four files: ```method-names.txt```, ```method-apis.txt```, ```method-tokens.txt```, and ```javadoc.txt```. These files are stored in the specified output directory. If no output directory is given, then files are stored in the ```data/``` folder. The folders ```train_data/```, ```train_data_smaller/```, ```validation_data/```, and ```validation_data_smaller/``` contain pre-generated datasets.

# Training Models
A new model can be trained using the command below.
```sh
python main.py --train --params <params-file> --train-dir <train-dir> --valid-dir <validation-dir>
```
The parameters input specifies a JSON file which contains the hyperparameters used for training. The folder ```params/``` contains existing hyperparameter files. Both the ``train-dir`` and ```valid-dir```  inputs are expected to hold datasets generated using the ```generate``` component described above. These parameters default to the directories ```train_data/``` and ```validation_data/``` respectively. Trained models are saved in a folder within the ```trained_models/``` directory under a name which specifies both the model type and the timestamp of training. A set of pre-trained models can be found [here](https://drive.google.com/drive/folders/17geATWd7CrF_XycpbYNQUBR4urU-pUvh?usp=sharing). The command below is a concrete example of a command to train a new model.
```sh
python main.py --train --params params/max_pool_rnn.json
```

# Indexing a Search Corpus
A corpus of Java code can be indexed into Redis using the command below.
```
python main.py --index --input <input-file-dir> --model <model-dir> --table <table-name>
```
The input is detected to be a directory if the string ends in a ```/```. Otherwise, the input is assumed to be the name of a single file. An input directory should contain protobuf files which encode the AST tree of Java classes.  The model directory specifies the trained model used to compute method embeddings. This command writes methods into the given table in a locally running Redis database. This command fails if a local Redis server is not running. If no table name is provided, methods are written into the table ```code```.

# Executing Searches
A search can be executed against an indexed corpus using the command below.
```
python main.py --search [--rerank] --input <query> --model <model-dir> --threshold <num-results> --table <table-name>
```
The given query can be either a string or a text file which contains a query on each line. The folder ```queries``` contains examples of query files sourced from StackOverflow. The outputs are written into the ```searches/<model-name>/``` directory as a formatted Java files. The specified model should be the same model used to index methods into the given table. The threshold parameter determines the number of results to return. The optional ```--rerank``` flag signals whether or not to re-rank outputs using the BM25F-based ranker.  A set of search results executed over the JSoup corpus can be found in the folder ```searches/jsoup```.
