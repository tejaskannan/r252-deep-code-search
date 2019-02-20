import numpy as np
import redis
import os
import heapq

from parser import Parser
from utils import cosine_similarity
from annoy import AnnoyIndex

REDIS_KEY_FORMAT = "{0}:{1}"
METHOD_NAME = "method_name"
METHOD_API = "method_api"
METHOD_BODY = "method_body"
METHOD_TOKENS = "method_tokens"
INDEX_PATH = "index/{0}_index.ann"

class DeepCodeSearchDB:

    def __init__(self, table, model, embedding_size, num_trees=16,
                 host="localhost", port=6379, pwd=0):
        self.redis_db = redis.Redis(host=host, port=port, db=pwd)
        self.data_table = table + "_data"
        self.emb_table = table + "_emb"
        self.index_path = INDEX_PATH.format(table)
        self.model = model
        self.embedding_size = embedding_size
        self.index = AnnoyIndex(embedding_size, metric="euclidean")
        self.num_trees = num_trees

    def index_dir(self, dir_name):
        method_id = 0
        count = 0
        for root, _dirs, files in os.walk(dir_name):
            for file_name in files:
                file_path = root + "/" + file_name

                method_id += self.index_file(file_path, method_id)
                count += 1
                
                if count % 100 == 0:
                    print("Indexed {0} files".format(count))

        self.index.build(self.num_trees)
        self.index.save(self.index_path)
        return method_id

    # Returns the number of methods indexed using this file
    def index_file(self, file_name, start_index=0):
        parser = Parser("filters/tags.txt", "filters/stopwords.txt")
        method_tokens, method_apis, method_names, javadocs, method_body = parser.parse_file(file_name)
        index = start_index
        for name, api, token, body in zip(method_names, method_apis, method_tokens, method_body):

            embedding = self.model.embed_method(name, api, token)

            data_key = REDIS_KEY_FORMAT.format(self.data_table, index)
            emb_key = REDIS_KEY_FORMAT.format(self.emb_table, index)

            self.redis_db.hset(data_key, METHOD_NAME, name)
            self.redis_db.hset(data_key, METHOD_API, api)
            self.redis_db.hset(data_key, METHOD_TOKENS, token)
            self.redis_db.hset(data_key, METHOD_BODY, body)

            # We have to explicitly delete any existing embedding vector because
            # inserting a vector amounts to appending to a redis list. Thus, plain insertions will
            # not overwrite eixisting data and instead would cause vectors to be erroneously large.
            if self.redis_db.exists(emb_key):
                self.redis_db.delete(emb_key)

            pipeline = self.redis_db.pipeline()
            for entry in embedding:
                pipeline.rpush(emb_key, str(entry))
            pipeline.execute()

            self.index.add_item(index, embedding)

            index += 1
        return index - start_index

    # K is the max number of results to return
    def search(self, description, k=10):

        def get_index(key):
            return key.split(":")[1].replace("'", "")

        embedded_descr = self.model.embed_description(description)
        top_results = []

        self.index.load(self.index_path)
        nearest_indices = self.index.get_nns_by_vector(embedded_descr, k)

        search_pipeline = self.redis_db.pipeline()
        for method_index in nearest_indices:
            key = REDIS_KEY_FORMAT.format(self.data_table, method_index)
            search_pipeline.hget(key, METHOD_BODY)

        results = []
        for method_body in search_pipeline.execute():
            results.append(method_body)
        return results


class Similarity:
    
    def __init__(self, redis_id, sim_score):
        self.redis_id = redis_id
        self.sim_score = sim_score

    def __eq__(self, other):
        return self.sim_score == other.sim_score

    def __ne__(self, other):
        return self.sim_score != other.sim_score

    def __lt__(self, other):
        return self.sim_score < other.sim_score

    def __gt__(self, other):
        return self.sim_score > other.sim_score

    def __le__(self, other):
        return self.sim_score <= other.sim_score

    def __ge__(self, other):
        return self.sim_score >= other.sim_score
