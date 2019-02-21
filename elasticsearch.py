import os
import json
import requests
from parser import Parser
from utils import write_to_file

INDEX_FORMAT = "{{\"index\": {{ \"_index\": \"{0}\", \"_type\": \"_doc\", \"_id\": \"{1}\" }} }}\n"
DATA_FORMAT = "{{\"method_body\": \"{0}\", \"method_name\": \"{1}\" }}\n"
THRESHOLD = 500

def create_data_files():
    dir_name = "../r252-corpus/r252-corpus-features/test/"
    output_file = "elasticsearch/data.json"
    index_name = "code"

    parser = Parser("filters/tags.txt", "filters/stopwords.txt")
    data_entries = []
    count = 0
    for root, _dirs, files in os.walk(dir_name):
            for file_name in files:
                file_path = root + "/" + file_name

                _t, _a, names, _j, method_body = parser.parse_file(file_path, only_javadoc=False)
                
                for i in range(len(method_body)):
                    name = names[i]
                    body = method_body[i].strip().replace("\n", "").replace("\"", "'")
                    data_entries.append(INDEX_FORMAT.format(index_name, count))
                    data_entries.append(DATA_FORMAT.format(body, name))
                    count += 1

                if len(data_entries) > THRESHOLD:
                    with open(output_file, "a") as out_file:
                        for entry in data_entries:
                            out_file.write(entry)
                    data_entries = []

def execute_queries():
    query_file_path = "queries/queries.txt"
    output_folder = "searches/elasticsearch/"
    headers = { "Content-Type": "application/json" }

    with open(query_file_path, "r") as query_file:
        for query in query_file:
            query_str = query.strip()

            tokens = query_str.split(" ")
            min_should_match = len(tokens) if len(tokens) < 2 else max(len(tokens) / 2, 2)

            query_obj = {
                "query": {
                    "multi_match": {
                        "query": query_str,
                        "analyzer": "method_analyzer",
                        "type": "cross_fields",
                        "fields": ["method_body", "method_name^3"],
                        "minimum_should_match": int(min_should_match) 
                    }
                }
            }

            data = json.dumps(query_obj)
            search = requests.get("http://localhost:9200/code/_search", headers=headers, data=data)
            resp_json = search.json()

            total = resp_json['hits']['total']
            if total == 0:
                continue

            hits = resp_json['hits']['hits']
            method_bodies = []
            for response in hits:
                method_body = response['_source']['method_body']
                method_bodies.append(method_body)

            write_to_file(output_folder + query_str.replace(" ", "_") + ".txt", method_bodies)

execute_queries()
