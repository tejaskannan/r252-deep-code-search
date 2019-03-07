import numpy as np
import redis
import os
import heapq

from parser import Parser
from utils import cosine_similarity, get_index, get_ranking_in_array, get_ranking_in_heap
from annoy import AnnoyIndex
from constants import *


class DeepCodeSearchDB:

    def __init__(self, table, model, embedding_size, num_trees=32,
                 host='localhost', port=6379, pwd=0):
        self.redis_db = redis.Redis(host=host, port=port, db=pwd)

        self.table = table
        self.data_table = table + '_data'
        self.emb_table = table + '_emb'
        self.freq_table = table + '_freq'
        self.num_docs_table = REDIS_KEY_FORMAT.format('num_docs', table)
        self.avg_doc_size_table = REDIS_KEY_FORMAT.format('avg_doc_size', table)

        self.index_path = INDEX_PATH.format(table)
        self.model = model
        self.embedding_size = embedding_size
        self.index = AnnoyIndex(embedding_size, metric='euclidean')
        self.num_trees = num_trees
        self.parser = Parser('filters/tags.txt', 'filters/stopwords.txt')

        self.k1 = 1.2
        self.b = 0.75

    def index_dir(self, dir_name, should_subtokenize=False):
        method_id = 0
        count = 0
        doc_counts = {}
        total_num_tokens = 0
        for root, _dirs, files in os.walk(dir_name):
            for file_name in files:
                file_path = root + '/' + file_name

                method_id, token_count = self.index_file(file_path, method_id, doc_counts=doc_counts,
                                                         should_subtokenize=should_subtokenize)
                count += 1
                total_num_tokens += token_count

                if count % REPORT_THRESHOLD == 0:
                    print('Indexed {0} files'.format(count))

        self.index.build(self.num_trees)
        self.index.save(self.index_path)

        # Write frequency scores
        freq_pipeline = self.redis_db.pipeline()
        for i, (token, doc_count) in enumerate(doc_counts.items()):
            freq_key = REDIS_KEY_FORMAT.format(self.freq_table, token)
            freq_pipeline.set(freq_key, str(doc_count))
            if i % REPORT_THRESHOLD:
                freq_pipeline.execute()
                freq_pipeline = self.redis_db.pipeline()

        freq_pipeline.set(self.num_docs_table, str(method_id))
        freq_pipeline.set(self.avg_doc_size_table, str(float(total_num_tokens) / method_id))
        freq_pipeline.execute()

        return method_id

    # Returns the number of methods indexed using this file
    def index_file(self, file_name, start_index=0, doc_counts={}, should_subtokenize=False):
        method_tokens, method_apis, method_names, _, method_body = self.parser.parse_file(file_name, only_javadoc=True,
                                                                                          should_subtokenize=should_subtokenize)
        index = start_index
        total_tokens = 0
        for name, api, token, body in zip(method_names, method_apis, method_tokens, method_body):

            embedding = self.model.embed_method(name, api, token)

            data_key = REDIS_KEY_FORMAT.format(self.data_table, index)
            emb_key = REDIS_KEY_FORMAT.format(self.emb_table, index)


            # Accumulate document counts for each token
            name_lst = name.split()
            api_lst = api.split()
            token_lst = token.split()
            all_tokens = name_lst + api_lst + token_lst
            for t in set(all_tokens):
                if t not in doc_counts:
                    doc_counts[t] = 0
                doc_counts[t] += 1

            total_tokens += len(all_tokens)

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
        return index, total_tokens

    # This function implements a baseline test where we use javadoc comments to search the
    # corpus and expect the corresponding method to be returned (using the full search strategy)
    # For correct results, the given directory should be the same as that of the indexed dataset.
    def hit_rank_over_corpus(self, corpus_dir, k=10):
        total_hit_rank = 0.0
        total_hits = 0.0
        total_queries = SMALL_NUMBER

        self.index.load(self.index_path)

        for root, _dirs, files in os.walk(corpus_dir):
            for file_name in files:
                file_path = root + '/' + file_name

                _t, _a, names, javadocs, _b = self.parser.parse_file(file_path, only_javadoc=True)
                for javadoc, name in zip(javadocs, names):

                    if len(javadoc) == 0:
                        continue

                    results = self.search(' '.join(javadoc), field=METHOD_NAME, k=k)         
                    hit_rank = get_ranking_in_array(results, name)

                    # This means that the method was found
                    if hit_rank != -1:
                        total_hits += 1.0
                        total_hit_rank += 1.0 / (hit_rank + 1)
                    total_queries += 1

        return total_hits / total_queries, total_hit_rank / total_queries

    # K is the max number of results to return
    # uses annoy for approximate (but faster) searching
    def search(self, description, field, k=10):
        description = self.parser.text_filter.apply_to_javadoc(description)
        embedded_descr = self.model.embed_description(description)
        top_results = []

        num_docs, avg_doc_size, freqs = self._fetch_frequencies(description)

        self.index.load(self.index_path)

        normalized_descr = embedded_descr / np.linalg.norm(embedded_descr)
        nearest_indices, nearest_scores = self.index.get_nns_by_vector(normalized_descr, 2*k,
                                                                       include_distances=True)

        search_pipeline = self.redis_db.pipeline()
        for method_index in nearest_indices:
            key = REDIS_KEY_FORMAT.format(self.data_table, method_index)
            search_pipeline.hgetall(key)

        redis_results = search_pipeline.execute()
        fetch_results = []
        for redis_hash in redis_results:
            decoded_hash = {key.decode('utf-8'): val.decode('utf-8') for key, val in redis_hash.items()}
            fetch_results.append(decoded_hash)

        # Re-rank outputs by accounting for bm25 lexical match scores
        results = []
        scores = []
        for dist_score, method in zip(nearest_scores, fetch_results):
            bm25_score = self._calculate_bm25(description=description,
                                              tokens=method[METHOD_TOKENS],
                                              apis=method[METHOD_API],
                                              name=method[METHOD_NAME],
                                              num_docs=num_docs,
                                              avg_doc_size=avg_doc_size,
                                              doc_freqs=freqs)
            score = bm25_score * dist_score
            results.append(method[field])
            scores.append(score)

        return [res for _,res in reversed(sorted(zip(scores, results)))][:k]

    # Searches for relevant answer by iterating over the entire dataset
    def search_full(self, description, k=10):
        description = self.parser.text_filter.apply_to_javadoc(description)
        top_results = self._search_full(description, k)

        results = []
        while len(top_results) > 0:
            result = heapq.heappop(top_results)
            key = REDIS_KEY_FORMAT.format(self.data_table, get_index(result[2]))
            method_body = self.redis_db.hget(key, METHOD_BODY)
            results.append(method_body)

        return results

    # Returns the top k results as a min PQ
    def _search_full(self, description, k):
        embedded_descr = self.model.embed_description(description)
        top_results = []

        counter = 0
        for key in self.redis_db.scan_iter(REDIS_KEY_FORMAT.format(self.emb_table, '*')):
            embedded_code = list(map(lambda x: float(str(x.decode('utf-8'))), self.redis_db.lrange(key, 0, -1)))
            sim_score = cosine_similarity(embedded_descr, embedded_code)
            heapq.heappush(top_results, (sim_score, counter, str(key.decode('utf-8'))))
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

    def _fetch_frequencies(self, description):
        num_docs = self.redis_db.get(self.num_docs_table)
        avg_doc_size = self.redis_db.get(self.avg_doc_size_table)

        freq_pipeline = self.redis_db.pipeline()
        for token in description:
            redis_key = REDIS_KEY_FORMAT.format(self.freq_table, token)
            freq_pipeline.get(redis_key)
        freq_results = freq_pipeline.execute()
        freq_results = [(int(f.decode('utf-8')) if f is not None else 0) for f in freq_results]

        freqs = {description[i]: freq_results[i] for i in range(len(description))}
        return int(num_docs.decode('utf-8')), float(avg_doc_size.decode('utf-8')), freqs

    def _calculate_bm25(self, description, tokens, apis, name, num_docs, avg_doc_size, doc_freqs):
        method = (tokens + apis + name).split()
        score = 0.0

        for query_token in description:

            count_in_method = float(np.sum([int(query_token == t) for t in method]))

            doc_freq = doc_freqs[query_token] if query_token in doc_freqs else 0
            idf = np.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            method_freq = (count_in_method) / len(method)

            numerator = (1 + self.k1) * method_freq
            denominator = method_freq + self.k1 * (1 - self.b + (len(method) / avg_doc_size))
            method_score = idf * (numerator / denominator)

            score += max(method_score, 0.0)
        return (1 + score)

