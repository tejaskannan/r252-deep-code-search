#!/usr/bin/python

import os

from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from text_filter import TextFilter


JAVADOC_FILE_NAME = "javadoc.txt"
METHOD_NAME_FILE_NAME = "method-names.txt"
METHOD_API_FILE_NAME = "method-apis.txt"
METHOD_TOKENS_FILE_NAME = "method-tokens.txt"

METHOD = "METHOD"

class Parser:

    def __init__(self, tags_file, keywords_file):
        self.text_filter = TextFilter(tags_file, keywords_file)

    def parse_directory(self, base, output_folder="data"):
        for root, _dirs, files in os.walk(base):
            for file_name in files:
                file_path = root + "/" + file_name
                self.parse_file(file_path)

    def parse_file(self, file_name, output_folder="data"):
        with open(file_name, 'rb') as proto_file:
            g = Graph()
            try:
                g.ParseFromString(proto_file.read())
            except:
                print("Error parsing: " + file_name)
                return

            javadoc_nodes = list(filter(lambda n: n.type == FeatureNode.COMMENT_JAVADOC, g.node))
            for javadoc_node in javadoc_nodes:
                javadoc_edge = next(e for e in g.edge if e.sourceId == javadoc_node.id)
                entity_node = next(n for n in g.node if javadoc_edge.destinationId == n.id)

                # We only consider methods in our dataset
                if entity_node.contents != METHOD:
                    continue

                method_text = self.extract_method_tokens(g,
                                                         entity_node.startLineNumber,
                                                         entity_node.endLineNumber)

                method_tokens_out_file = output_folder + "/" + METHOD_TOKENS_FILE_NAME
                cleaned_method = self.clean_text(method_text)
                self.append_to_file(cleaned_method, method_tokens_out_file)

                javadoc_out_file = output_folder + "/" + JAVADOC_FILE_NAME
                cleaned_javadoc = self.clean_javadoc(javadoc_node.contents)
                self.append_to_file(cleaned_javadoc, javadoc_out_file)


    def extract_method_tokens(self, graph, start, end):

        def is_method_token(node):
            return (node.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN)) and \
                   (node.startLineNumber >= start) and \
                   (node.endLineNumber <= end)

        method_nodes = list(filter(is_method_token, graph.node))
        return " ".join(list(map(lambda n: n.contents, method_nodes)))

    def clean_text(self, text):
        text = text.lower()
        tokens = text.split()
        tokens = self.text_filter.apply_to_token_lst(tokens)
        return " ".join(tokens)

    def clean_javadoc(self, javadoc):
        javadoc_contents = javadoc.replace("/", "").replace("*", "") \
                                  .replace("{", "").replace("}", "")
        cleaned_javadoc = self.text_filter.apply_to_javadoc(javadoc_contents)
        return " ".join(cleaned_javadoc)

    def append_to_file(self, text, file_name):
        with open(file_name, "a") as file:
            file.write(text + "\n")
