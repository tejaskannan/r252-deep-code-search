import os
import json
import requests
import subprocess
import time
import numpy as np
from parser import Parser
from utils import write_methods_to_file, get_ranking_in_array

INDEX_FORMAT = '{{\"index\": {{ \"_index\": \"{0}\", \"_type\": \"_doc\", \"_id\": \"{1}\" }} }}\n'
DATA_FORMAT = '{{\"method_body\": \"{0}\", \"method_name\": \"{1}\" }}\n'
THRESHOLD = 500

def create_data_files():
    dir_name = '../r252-corpus/r252-corpus-features/validation/'
    output_file = 'elasticsearch/data.json'
    index_name = 'code'

    parser = Parser('filters/tags.txt', 'filters/stopwords.txt')
    data_entries = []
    count = 0
    for root, _dirs, files in os.walk(dir_name):
        for file_name in files:
            file_path = root + '/' + file_name

            _t, _a, names, _j, method_body = parser.parse_file(file_path, only_javadoc=False)

            for i in range(len(method_body)):
                name = names[i]
                body = method_body[i].strip().replace('\"', '\\\"').replace('\'', '\\\'')
                data_entries.append(INDEX_FORMAT.format(index_name, count))
                data_entries.append(DATA_FORMAT.format(body, name))
                count += 1

            if len(data_entries) > THRESHOLD:
                with open(output_file, 'a') as out_file:
                    for entry in data_entries:
                        out_file.write(entry)
                data_entries = []


def execute_javadoc_queries():
    dir_name = '../r252-corpus/r252-corpus-features/validation/'
    output_file = 'elasticsearch/data.json'
    index_name = 'code'

    parser = Parser('filters/tags.txt', 'filters/stopwords.txt')
    data_entries = []

    total_hit_rank = 0.0
    total_hits = 0.0
    total_queries = 0.0

    for root, _dirs, files in os.walk(dir_name):
        for file_name in files:
            file_path = root + '/' + file_name

            _t, _a, names, javadocs, _b = parser.parse_file(file_path, only_javadoc=True)

            for name, javadoc in zip(names, javadocs):
                method_names = execute_query(javadoc)[1]

                hit_rank = get_ranking_in_array(method_names, name)
                if hit_rank != -1:
                    total_hit_rank += 1.0 / float(hit_rank)
                    total_hits += 1
                total_queries += 1

    print('Success Rate: {0}'.format(total_hits / total_queries))
    print('MRR: {0}'.format(total_hit_rank / total_queries))            


def execute_queries():
    query_file_path = 'queries/jsoup_queries.txt'
    output_folder = 'searches/elasticsearch/'
    formatter = 'formatter/google-java-format-1.7-all-deps.jar'

    with open(query_file_path, 'r') as query_file:
        times = []
        for query in query_file:
            query_str = query.strip()

            start = time.time()
            method_bodies = execute_query(query_str)[0]
            elapsed = time.time() - start
            times.append(elapsed)

            out_base = output_folder + query_str.replace(' ', '_')
            out_path_temp = out_base + '-temp.java'
            write_methods_to_file(out_path_temp, method_bodies)

            out_path = out_base + '.java'
            with open(out_path, 'w') as out_file:
                subprocess.call(['java', '-jar', formatter, out_path_temp], stdout=out_file)
        times.pop(0)
        print('Average Query Time: {0}s'.format(np.average(times)))
        print('Std Query Time: {0}'.format(np.std(times)))

def execute_query(query_str):
    headers = {'Content-Type': 'application/json'}

    num_tokens = len(query_str.split())
    min_should_match = num_tokens if num_tokens < 2 else min(num_tokens / 2, 2)

    query_obj = {
        'query': {
            'multi_match': {
                'query': query_str,
                'analyzer': 'method_analyzer',
                'type': 'cross_fields',
                'fields': ['method_body', 'method_name^2'],
            }
        }
    }

    data = json.dumps(query_obj)
    search = requests.get('http://localhost:9200/code/_search', headers=headers, data=data)
    resp_json = search.json()

    total = resp_json['hits']['total']
    if total == 0:
        return [], []

    hits = resp_json['hits']['hits']
    method_bodies = []
    method_names = []
    for response in hits:
        method_body = response['_source']['method_body']
        method_name = response['_source']['method_name']
        method_bodies.append(method_body)
        method_names.append(method_name)
    return method_bodies, method_names


execute_queries()
