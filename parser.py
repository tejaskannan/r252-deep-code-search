#!/usr/bin/python

import os
import re

from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from text_filter import TextFilter
from code_graph import CodeGraph


JAVADOC_FILE_NAME = "javadoc.txt"
METHOD_NAME_FILE_NAME = "method-names.txt"
METHOD_API_FILE_NAME = "method-apis.txt"
METHOD_TOKENS_FILE_NAME = "method-tokens.txt"

METHOD = "METHOD"
DOT = "DOT"
LPAREN = "LPAREN"
VARIABLE = "VARIABLE"
TYPE = "TYPE"

class Parser:

    def __init__(self, tags_file, keywords_file):
        self.text_filter = TextFilter(tags_file, keywords_file)

    def parse_directory(self, base, output_folder="data"):
        for root, _dirs, files in os.walk(base):
            for file_name in files:
                file_path = root + "/" + file_name
                self.parse_file(file_path)

    def parse_file(self, file_name, output_folder="data"):

        method_name_file = output_folder + "/" + METHOD_NAME_FILE_NAME
        method_api_file = output_folder + "/" + METHOD_API_FILE_NAME
        method_tokens_file = output_folder + "/" + METHOD_TOKENS_FILE_NAME
        javadoc_tokens_file = output_folder + "/" + JAVADOC_FILE_NAME

        with open(file_name, 'rb') as proto_file:
            g = Graph()
            try:
                g.ParseFromString(proto_file.read())
            except:
                print("Error parsing: " + file_name)
                return

            code_graph = CodeGraph(g)

            # javadoc_nodes = list(filter(lambda n: n.type == FeatureNode.COMMENT_JAVADOC, g.node))
            # for javadoc_node in javadoc_nodes:
            #     javadoc_edge = next(e for e in g.edge if e.sourceId == javadoc_node.id)
            #     entity_node = next(n for n in g.node if javadoc_edge.destinationId == n.id)

            #     # We only consider methods in our dataset
            #     if entity_node.contents != METHOD:
            #         continue

            #     javadoc_out_file = output_folder + "/" + JAVADOC_FILE_NAME
            #     cleaned_javadoc = self.clean_javadoc(javadoc_node.contents)
            #     if len(cleaned_javadoc) == 0:
            #         continue
            #     self.append_to_file(cleaned_javadoc, javadoc_out_file)

            #     method_name, method_api_calls, method_tokens = self.extract_method_tokens(
            #                     g, entity_node.startLineNumber, entity_node.endLineNumber)

            #     self.append_to_file(" ".join(method_name), method_name_file)
            #     self.append_to_file(" ".join(method_api_calls), method_api_file)

            #     cleaned_tokens = self.clean_tokens(method_tokens)
            #     self.append_to_file(cleaned_tokens, method_tokens_file)


    def extract_method_tokens(self, graph, start, end):

        def is_method_token(node):
            return (node.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN)) and \
                   (node.startLineNumber >= start) and \
                   (node.endLineNumber <= end)

        method_nodes = list(filter(is_method_token, graph.node))

        method_name = []
        method_api_calls = []
        method_tokens = []

        first = True
        for i in range(0, len(method_nodes)):
            node = method_nodes[i]
            if node.type == FeatureNode.TOKEN:
                continue
            if first:
                method_name = self.split_camel_case(node.contents)
                method_name = list(map(lambda s: s.lower(), method_name))
                first = False
            elif (i+1) < len(method_nodes) and method_nodes[i+1].type == FeatureNode.TOKEN and \
                 (method_nodes[i+1].contents == DOT or method_nodes[i+1].contents == LPAREN):
                api_tokens = self.split_camel_case(node.contents)
                method_api_calls += list(map(lambda s: s.lower(), api_tokens))
            else:
                method_tokens.append(node.contents.lower())

        return method_name, method_api_calls, method_tokens

    def clean_tokens(self, tokens):
        return " ".join(self.text_filter.apply_to_token_lst(tokens))

    def clean_javadoc(self, javadoc):
        javadoc_contents = javadoc.replace("/", "").replace("*", "") \
                                  .replace("{", "").replace("}", "").lower()
        cleaned_javadoc = self.text_filter.apply_to_javadoc(javadoc_contents)
        return " ".join(cleaned_javadoc)

    def append_to_file(self, text, file_name):
        with open(file_name, "a") as file:
            file.write(text + "\n")

    def split_camel_case(self, text):
        return re.sub('([a-z])([A-Z])', r'\1 \2', text).split()




class Variable:

    def __init__(self, type_node, var_node):
        self.type_node = type_node
        self.var_node = var_node