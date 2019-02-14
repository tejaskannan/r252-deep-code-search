import numpy as np
import redis
import os

from parser import Parser
import heapq
from utils import cosine_similarity

REDIS_KEY_FORMAT = "{0}:{1}"
METHOD_NAME = "method_name"
METHOD_API = "method_api"
METHOD_BODY = "method_body"
METHOD_TOKENS = "method_tokens"

class DeepCodeSearchDB:

    def __init__(self, table, model, host="localhost", port=6379, pwd=0):
        self.redis_db = redis.Redis(host=host, port=port, db=pwd)
        self.data_table = table + "_data"
        self.emb_table = table + "_emb"
        self.model = model

    # Since embedding vectors are stored as lists, the code is really only designed to
    # index all at once. Indexing multiple times will cause vectors to be written
    # to the same redis list. This will NOT overwrite existing information--instead, the
    # lists will be appended with more elements. This causes the cosine similarity metric to fail.
    def index_dir(self, dir_name, start_index=0):
        index = start_index
        for root, _dirs, files in os.walk(dir_name):
            for file_name in files:
                file_path = root + "/" + file_name
                index += self.index_file(file_path, index)
        return index - start_index

    # Returns the number of methods indexed using this file
    def index_file(self, file_name, start_index=0):
        parser = Parser("filters/tags.txt", "filters/stopwords.txt")
        method_names, method_apis, method_tokens, javadocs, method_body = parser.parse_file(file_name)
        index = start_index
        for name, api, token, body in zip(method_names, method_apis, method_tokens, method_body):
            embedding = self.model.embed_method(name, api, token)

            data_key = REDIS_KEY_FORMAT.format(self.data_table, index)
            emb_key = REDIS_KEY_FORMAT.format(self.emb_table, index)

            self.redis_db.hset(data_key, METHOD_NAME, name)
            self.redis_db.hset(data_key, METHOD_API, api)
            self.redis_db.hset(data_key, METHOD_TOKENS, token)
            self.redis_db.hset(data_key, METHOD_BODY, body)

            for entry in embedding:
                self.redis_db.rpush(emb_key, str(entry))

            index += 1
        return index - start_index

    # K is the max number of results to return
    def search(self, description, k=10):

        def get_index(key):
            return key.split(":")[1].replace("'", "")

        embedded_descr = self.model.embed_description(description)
        top_results = []

        for key in self.redis_db.scan_iter(REDIS_KEY_FORMAT.format(self.emb_table, "*")):
            embedded_code = list(map(lambda x: float(x), self.redis_db.lrange(key, 0, -1)))
            sim_score = cosine_similarity(embedded_descr, embedded_code)
            heapq.heappush(top_results, SimScore(str(key), sim_score))
            if len(top_results) > k:
                heapq.heappop(top_results)

        results = []
        for result in reversed(top_results):
            key = REDIS_KEY_FORMAT.format(self.data_table, get_index(result.redis_id))
            method_body = self.redis_db.hget(key, METHOD_BODY)
            results.append(method_body)

        return results

class SimScore:
    
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
