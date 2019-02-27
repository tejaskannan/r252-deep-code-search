import numpy as np
import redis
import os
import heapq

from parser import Parser
from utils import cosine_similarity, get_index
from annoy import AnnoyIndex
from constants import *


class DeepCodeSearchDB:

    def __init__(self, table, model, embedding_size, num_trees=32,
                 host='localhost', port=6379, pwd=0):
        self.redis_db = redis.Redis(host=host, port=port, db=pwd)
        self.data_table = table + '_data'
        self.emb_table = table + '_emb'
        self.index_path = INDEX_PATH.format(table)
        self.model = model
        self.embedding_size = embedding_size
        self.index = AnnoyIndex(embedding_size, metric='dot')
        self.num_trees = num_trees

    def index_dir(self, dir_name):
        method_id = 0
        count = 0
        for root, _dirs, files in os.walk(dir_name):
            for file_name in files:
                file_path = root + '/' + file_name

                method_id = self.index_file(file_path, method_id)
                count += 1

                if count % REPORT_THRESHOLD == 0:
                    print('Indexed {0} files'.format(count))

        self.index.build(self.num_trees)
        self.index.save(self.index_path)
        return method_id

    # Returns the number of methods indexed using this file
    def index_file(self, file_name, start_index=0):
        parser = Parser('filters/tags.txt', 'filters/stopwords.txt')
        method_tokens, method_apis, method_names, _, method_body = parser.parse_file(file_name, only_javadoc=False)
        index = start_index
        for name, api, token, body in zip(method_names, method_apis, method_tokens, method_body):

            embedding = self.model.embed_method(name, api, token)

            data_key = REDIS_KEY_FORMAT.format(self.data_table, index)
            emb_key = REDIS_KEY_FORMAT.format(self.emb_table, index)

            pipeline = self.redis_db.pipeline()

            pipeline.hset(data_key, METHOD_NAME, name)
            pipeline.hset(data_key, METHOD_API, api)
            pipeline.hset(data_key, METHOD_TOKENS, token)
            pipeline.hset(data_key, METHOD_BODY, body)

            # We have to explicitly delete any existing embedding vector because
            # inserting a vector amounts to appending to a redis list. Simply inserting data will
            # not overwrite existing entries and instead cause vectors to be erroneously large.
            if self.redis_db.exists(emb_key):
                self.redis_db.delete(emb_key)

            for entry in embedding:
                pipeline.rpush(emb_key, str(entry))
            pipeline.execute()

            normalized_embedding = embedding / np.linalg.norm(embedding)
            self.index.add_item(index, normalized_embedding)

            index += 1
        return index

    # K is the max number of results to return
    # uses annoy for approximate (but faster) searching
    def search(self, description, k=10):

        embedded_descr = self.model.embed_description(description)
        top_results = []

        self.index.load(self.index_path)

        normalized_descr = embedded_descr / np.linalg.norm(embedded_descr)
        nearest_indices = self.index.get_nns_by_vector(normalized_descr, k)

        search_pipeline = self.redis_db.pipeline()
        for method_index in nearest_indices:
            key = REDIS_KEY_FORMAT.format(self.data_table, method_index)
            search_pipeline.hget(key, METHOD_BODY)

        results = []
        for method_body in search_pipeline.execute():
            results.append(method_body)
        return results

    # Searches for relevant answer by iterating over the entire dataset
    def search_full(self, description, k=10):

        embedded_descr = self.model.embed_description(description)
        normalized_descr = embedded_descr / np.linalg.norm(embedded_descr)
        top_results = []

        counter = 0
        for key in self.redis_db.scan_iter(REDIS_KEY_FORMAT.format(self.emb_table, '*')):
            embedded_code = list(map(lambda x: float(str(x.decode('utf-8'))), self.redis_db.lrange(key, 0, -1)))
            sim_score = cosine_similarity(normalized_descr, embedded_code)
            heapq.heappush(top_results, (sim_score, counter, str(key.decode('utf-8'))))
            if len(top_results) > k:
                heapq.heappop(top_results)
            counter += 1

        results = []
        scores = []
        while len(top_results) > 0:
            result = heapq.heappop(top_results)
            scores.append(result[0])
            key = REDIS_KEY_FORMAT.format(self.data_table, get_index(result[2]))
            method_body = self.redis_db.hget(key, METHOD_BODY)
            results.append(method_body)

        return list(reversed(results))
