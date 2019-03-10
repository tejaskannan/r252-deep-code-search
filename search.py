import numpy as np
import redis
import os
import heapq

from parser import Parser
from utils import cosine_similarity, get_index, get_ranking_in_array, get_ranking_in_heap
from annoy import AnnoyIndex
from constants import *


class DeepCodeSearchDB:

    def __init__(self, table, model, embedding_size, num_trees=100,
                 host='localhost', port=6379, pwd=0):
        self.redis_db = redis.Redis(host=host, port=port, db=pwd)

        self.table = table
        self.data_table = table + '_data'
        self.emb_table = table + '_emb'

        # Redis K-V Pairs used for BM25F Re-Ranking
        self.freq_table = table + '_freq'
        self.num_docs_table = REDIS_KEY_FORMAT.format('num_docs', table)
        self.avg_length_table = REDIS_KEY_FORMAT.format('avg_len', table)

        self.index_path = INDEX_PATH.format(table)
        self.model = model
        self.embedding_size = embedding_size
        self.index = AnnoyIndex(embedding_size, metric='dot')
        self.num_trees = num_trees
        self.parser = Parser('filters/tags.txt', 'filters/stopwords.txt')

        self.k1 = 1.2
        self.b = 0.75
        self.name_weight = 1.0
        self.api_weight = 0.5
        self.token_weight = 0.25

    def index_dir(self, dir_name, should_subtokenize=False):
        method_id = 0
        count = 0

        # Dictionary to track term counts for later re-ranking
        doc_counts = {}
        doc_lengths = {
            METHOD_NAME: 0.0,
            METHOD_API: 0.0,
            METHOD_TOKENS: 0.0,
        }

        total_num_tokens = 0
        for root, _dirs, files in os.walk(dir_name):
            for file_name in files:
                file_path = root + '/' + file_name

                method_id = self.index_file(file_path, method_id, doc_counts=doc_counts,
                                            doc_lengths=doc_lengths,
                                            should_subtokenize=should_subtokenize)
                count += 1

                if count % REPORT_THRESHOLD == 0:
                    print('Indexed {0} files'.format(count))

        self.index.build(self.num_trees)
        self.index.save(self.index_path)

        # Write term counts
        self._write_counts_to_redis(doc_counts)

        # Write average field lengths
        pipeline = self.redis_db.pipeline()
        pipeline.hset(self.avg_length_table, METHOD_NAME, str(doc_lengths[METHOD_NAME] / method_id))
        pipeline.hset(self.avg_length_table, METHOD_API, str(doc_lengths[METHOD_API] / method_id))
        pipeline.hset(self.avg_length_table, METHOD_TOKENS, str(doc_lengths[METHOD_TOKENS] / method_id))
        pipeline.set(self.num_docs_table, str(method_id))
        pipeline.execute()

        return method_id

    # Returns the number of methods indexed using this file
    def index_file(self, file_name, start_index=0, doc_counts={}, doc_lengths={}, should_subtokenize=False):
        method_tokens, method_apis, method_names, _, method_body = self.parser.parse_file(file_name, only_javadoc=False,
                                                                                          should_subtokenize=should_subtokenize)
        pipeline = self.redis_db.pipeline()

        index = start_index
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

            # Track field lengths for later re-ranking
            if METHOD_NAME in doc_lengths:
                doc_lengths[METHOD_NAME] += len(name_lst)
            if METHOD_API in doc_lengths:
                doc_lengths[METHOD_API] += len(api_lst)
            if METHOD_TOKENS in doc_lengths:
                doc_lengths[METHOD_TOKENS] += len(token_lst)

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

            normalized_embedding = embedding / np.linalg.norm(embedding)
            self.index.add_item(index, normalized_embedding)

            index += 1

        pipeline.execute()
        return index

    # This function implements a baseline test where we use javadoc comments to search the
    # corpus and expect the corresponding method to be returned (using the full search strategy)
    # For correct results, the given directory should be the same as that of the indexed dataset.
    def hit_rank_over_corpus(self, corpus_dir, thresholds=[10], should_rerank=False):
        total_hit_ranks = np.zeros_like(thresholds, dtype=float)
        total_hits = np.zeros_like(thresholds, dtype=float)
        total_queries = SMALL_NUMBER

        self.index.load(self.index_path)

        for root, _dirs, files in os.walk(corpus_dir):
            for file_name in files:
                file_path = root + '/' + file_name

                _t, _a, names, javadocs, _b = self.parser.parse_file(file_path, only_javadoc=True)
                for javadoc, name in zip(javadocs, names):

                    if len(javadoc) == 0:
                        continue

                    javadoc_str = ' '.join(javadoc)
                    for i, k in enumerate(thresholds):
                        results = self.search(javadoc_str, field=METHOD_NAME, k=k,
                                              should_rerank=should_rerank)         
                        hit_rank = get_ranking_in_array(results, name)

                        if hit_rank != -1:
                            total_hits[i] += 1.0
                            total_hit_ranks[i] += 1.0 / hit_rank

                    total_queries += 1.0

        return total_hits / total_queries, total_hit_ranks / total_queries

    # K is the max number of results to return
    # uses annoy for approximate (but faster) searching
    def search(self, description, field, k=10, should_rerank=False):
        description = self.parser.text_filter.apply_to_javadoc(description)
        embedded_descr = self.model.embed_description(description)

        num_docs, avg_lengths, freqs = self._fetch_frequencies(description)

        self.index.load(self.index_path)

        normalized_descr = embedded_descr / np.linalg.norm(embedded_descr)
        nearest_indices, distances = self.index.get_nns_by_vector(normalized_descr, 2*k,
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

        results = [method[field] for method in fetch_results]
        if not should_rerank:
            return results[:k]

        # Re-rank outputs by accounting for bm25 lexical match scores
        bm25_scores = []
        for method in fetch_results:
            bm25_score = self._calculate_bm25f(description=description,
                                               tokens=method[METHOD_TOKENS],
                                               apis=method[METHOD_API],
                                               name=method[METHOD_NAME],
                                               avg_field_lengths=avg_lengths,
                                               num_docs=num_docs,
                                               doc_freqs=freqs)
            bm25_scores.append(bm25_score)

        exp_distances = np.exp(distances)
        bm25_scores = np.log(np.array(bm25_scores) + 1.0) + 1.0

        scores = np.multiply(exp_distances, bm25_scores)

        return [res for _,res in reversed(sorted(zip(scores, results)))][:k]

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
        pipeline = self.redis_db.pipeline()

        pipeline.hgetall(self.avg_length_table)
        pipeline.get(self.num_docs_table)

        for token in description:
            redis_key = REDIS_KEY_FORMAT.format(self.freq_table, token)
            pipeline.get(redis_key)
        results = pipeline.execute()

        avg_field_lengths = {name.decode('utf-8'): float(a.decode('utf-8')) for name, a in \
                             results[0].items()}

        num_docs = int(results[1].decode('utf-8'))

        freq_results = [(int(f.decode('utf-8')) if f is not None else 0) for f in results[2:]]
        freqs = {description[i]: freq_results[i] for i in range(len(description))}

        return num_docs, avg_field_lengths, freqs

    def _calculate_bm25f(self, description, tokens, apis, name, num_docs, avg_field_lengths, doc_freqs):
        score = 0.0

        name_lst = name.split()
        api_lst = apis.split()
        token_lst = tokens.split()

        for query_token in description:

            name_term_freq = self._calculate_term_freq_field(query_token, name_lst,
                                                             avg_field_lengths[METHOD_NAME])
            api_term_freq = self._calculate_term_freq_field(query_token, api_lst,
                                                            avg_field_lengths[METHOD_API])
            token_term_freq = self._calculate_term_freq_field(query_token, token_lst,
                                                              avg_field_lengths[METHOD_TOKENS]) 

            doc_freq = doc_freqs[query_token] if query_token in doc_freqs else 0.0
            idf = np.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

            term_score = self.name_weight * name_term_freq + self.api_weight * api_term_freq + \
                         self.token_weight * token_term_freq
            method_score = idf * term_score

            score += max(method_score, 0.0)

        return score

    def _calculate_term_freq_field(self, term, field_tokens, avg_field_len):
        count_in_field = float(np.sum([int(term == t) for t in field_tokens]))
        return count_in_field / (1 + self.b * ((len(field_tokens) / avg_field_len) - 1))

    def _write_counts_to_redis(self, doc_counts):
        freq_pipeline = self.redis_db.pipeline()

        for i, (token, doc_count) in enumerate(doc_counts.items()):
            freq_key = REDIS_KEY_FORMAT.format(self.freq_table, token)
            freq_pipeline.set(freq_key, str(doc_count))
            if i % REPORT_THRESHOLD:
                freq_pipeline.execute()
                freq_pipeline = self.redis_db.pipeline()
        freq_pipeline.execute()