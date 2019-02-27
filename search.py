import numpy as np
import redis
import os
import heapq

from parser import Parser
from utils import cosine_similarity, get_index, get_ranking
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
        self.parser = Parser('filters/tags.txt', 'filters/stopwords.txt')

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
        method_tokens, method_apis, method_names, _, method_body = self.parser.parse_file(file_name, only_javadoc=False)
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

    # This function implements a baseline test where we use javadoc comments to search the
    # corpus and expect the corresponding method to be returned (using the full search strategy)
    # For correct results, the given directory should be the same as that of the indexed dataset.
    def hit_rank_over_corpus(self, corpus_dir, k=10):
        total_hit_rank = 0.0
        total_hits = 0.0
        total_queries = 0.0

        method_id = 0
        for root, _dirs, files in os.walk(corpus_dir):
            for file_name in files:
                file_path = root + '/' + file_name

                _t, _a, names, javadocs, _b = self.parser.parse_file(file_path, only_javadoc=False)
                for javadoc, name in zip(javadocs, names):

                    if len(javadoc) == 0:
                        method_id += 1
                        continue

                    total_queries += 1

                    search_results = self._search_full(javadoc, k)
                    hit_rank = get_ranking(search_results, method_id)

                    method_id += 1

                    # This means that the method was  found
                    if hit_rank != -1:
                        total_hits += 1.0
                        total_hit_rank += 1.0 / (hit_rank + 1)  

        return total_hits / total_queries, total_hit_rank / total_queries

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

        top_results = self._search_full(description, k)

        results = []
        while len(top_results) > 0:
            result = heapq.heappop(top_results)
            key = REDIS_KEY_FORMAT.format(self.data_table, get_index(result[2]))
            method_body = self.redis_db.hget(key, METHOD_BODY)
            results.append(method_body)

        return results

    def _search_full(self, description, k):
        embedded_descr = self.model.embed_description(description)
        top_results = []

        counter = 0
        for key in self.redis_db.scan_iter(REDIS_KEY_FORMAT.format(self.emb_table, '*')):
            embedded_code = list(map(lambda x: float(str(x.decode('utf-8'))), self.redis_db.lrange(key, 0, -1)))
            sim_score = cosine_similarity(embedded_descr, embedded_code)
            heapq.heappush(top_results, (-sim_score, counter, str(key.decode('utf-8'))))
            if len(top_results) > k:
                heapq.heappop(top_results)
            counter += 1
        return top_results

    def vocabulary_overlap(self, dir_path):
        # In the order [token, api, name]
        overlap = [0, 0, 0]
        total = [0, 0, 0]

        for root, _dirs, files in os.walk(dir_path):
            for file_name in files:
                file_path = root + '/' + file_name

                tokens, apis, names, _javadoc, _body = self.parser.parse_file(file_path, only_javadoc=False)

                for i, token_lst in enumerate([tokens, apis, names]):
                    t, o = self._find_overlap(token_lst)
                    overlap[i] += o
                    total[i] += t

        return overlap, total

    def _find_overlap(self, tokens):
        overlap = 0
        total = 0
        for token_sequence in tokens:
            for token in token_sequence.split(' '):
                if not self.model.dataset.vocabulary.is_unk(token):
                    overlap += 1
                total += 1
        return total, overlap
