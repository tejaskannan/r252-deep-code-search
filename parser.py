#!/usr/bin/python

import os
import re

from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
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
METHOD_SELECT = "METHOD_SELECT"
MEMBER_SELECT = "MEMBER_SELECT"
EXPRESSION = "EXPRESSION"
STATEMENTS = "STATEMENTS"
METHOD_INVOCATION = "METHOD_INVOCATION"
LBRACE = "LBRACE"
NEW = "NEW"
NEW_LOWER = "new"
API_FORMAT = "{0}.{1}"

class Parser:

    def __init__(self, tags_file, keywords_file, line_threshold=3):
        self.text_filter = TextFilter(tags_file, keywords_file)
        self.line_threshold = line_threshold

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

            for method in code_graph.methods.values():

                # We skip very small methods because these can often be described with
                # heuristics
                # if method.num_lines <= self.line_threshold:
                #     continue

                method_name_tokens = self._split_camel_case(method.method_name)

                method_invocations = self._get_method_invocations(method.method_block, code_graph)
                api_call_tokens = []
                for invocation in method_invocations:
                    api_call_tokens.append(self._parse_method_invocation(invocation, code_graph))

                obj_init_tokens = self._get_object_inits(method.method_block, code_graph)
                api_call_tokens += obj_init_tokens

                javadoc_tokens = self._clean_javadoc(method.javadoc.contents)

                method_tokens = self._get_method_tokens(method.method_block, code_graph)
                method_tokens = self._clean_tokens(method_tokens)

                print(api_call_tokens)
                print(method_tokens)

            # method_invocations = code_graph.get_nodes_with_content("METHOD_INVOCATION")
            # for method_invoke in method_invocations:
            #     print(self._parse_method_invocation(method_invoke, code_graph))

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



    def _parse_method_invocation(self, method_invocation_node, code_graph):
        method_select = code_graph.get_neighbors_with_type_content(method_invocation_node.id,
                                                                   neigh_type=None,
                                                                   neigh_content=METHOD_SELECT)[0]
        member_select = code_graph.get_neighbors_with_type_content(method_select.id,
                                                                   neigh_type=None,
                                                                   neigh_content=MEMBER_SELECT)[0]
        
        expr_node = code_graph.get_neighbors_with_type_content(member_select.id,
                                                               neigh_type=FeatureNode.FAKE_AST,
                                                               neigh_content=EXPRESSION)[0]
        identifier = code_graph.get_neighbors_with_type_content(member_select.id,
                                                                neigh_type=FeatureNode.IDENTIFIER_TOKEN,
                                                                neigh_content=None)[0]

        token_node = code_graph.get_out_neighbors_with_edge_type(expr_node.id, FeatureEdge.ASSOCIATED_TOKEN)

        # This means we have a nested method call
        api_str = ""
        if token_node == None:
            inner_invocation = code_graph.get_out_neighbors_with_edge_type(expr_node.id, FeatureEdge.AST_CHILD)[0]
            api_str = self._parse_method_invocation(inner_invocation, code_graph)
        else:
            token = token_node[0]
            type_node = self._find_variable_type(token, code_graph)
            api_str = type_node.contents
        api_str = API_FORMAT.format(api_str, identifier.contents)
        return api_str

    def _find_variable_type(self, variable_node, code_graph):
        node = variable_node
        # We first move forward
        while node:
            if node.id in code_graph.vars:
                variable = code_graph.vars[node.id]
                return variable.type_node
            next_use = code_graph.get_out_neighbors_with_edge_type(node.id,
                                                                   FeatureEdge.LAST_LEXICAL_USE)
            if next_use:
                node = next_use[0]
            else:
                node = next_use

        node = variable_node
        while node:
            if node.id in code_graph.vars:
                variable = code_graph.vars[node.id]
                return variable.type_node
            prev_use = code_graph.get_in_neighbors_with_edge_type(node.id,
                                                                  FeatureEdge.LAST_LEXICAL_USE)
            if prev_use:
                node = prev_use[0]
            else:
                node = prev_use

        return None

    # This method only gets top-level invocations. Nested method calls are handled
    # separately in the parsing stage
    def _get_method_invocations(self, method_block, code_graph):
        statements = code_graph.get_neighbors_with_type_content(method_block.id,
                                                                neigh_type=FeatureNode.FAKE_AST,
                                                                neigh_content=STATEMENTS)[0]
        bounds = code_graph.get_neighbors_with_type_content(method_block.id,
                                                            neigh_type=FeatureNode.TOKEN,
                                                            neigh_content=None)
        method_invocations = []
        node_stack = [statements]

        # This initalization prevents an infinite loop by bounding
        # the search space
        seen_ids = { bounds[0].id, bounds[1].id }
        while len(node_stack) > 0:
            node = node_stack.pop()
            seen_ids.add(node.id)
            if node.contents == METHOD_INVOCATION:
                method_invocations.append(node)
            else:
                neighbors = code_graph.get_out_neighbors(node.id)
                for n in neighbors:
                    if not (n.id in seen_ids):
                        node_stack.append(n)
        return method_invocations


    def _get_method_tokens(self, method_block, code_graph):
        start, end = self._get_bounds(method_block, code_graph)

        tokens = []
        node = start
        while (node.id != end.id):
            if node.type == FeatureNode.IDENTIFIER_TOKEN:
                in_neighbors = code_graph.get_in_neighbors_with_edge_type(node.id, FeatureEdge.ASSOCIATED_TOKEN)
                for n in in_neighbors:
                    if not n.contents in (EXPRESSION, MEMBER_SELECT):
                        tokens += [token.lower() for token in self._split_camel_case(node.contents)]
            else:
                tokens.append(node.contents.lower())
            node = code_graph.get_out_neighbors_with_edge_type(node.id, FeatureEdge.NEXT_TOKEN)[0]
        return list(set(tokens))

    def _get_object_inits(self, method_block, code_graph):
        start, end = self._get_bounds(method_block, code_graph)

        tokens = []
        node = start
        while (node.id != end.id):
            if node.contents == NEW:
                obj_type_node = code_graph.get_out_neighbors_with_edge_type(node.id, FeatureEdge.NEXT_TOKEN)[0]
                tokens.append(API_FORMAT.format(obj_type_node.contents, NEW_LOWER))
                node = obj_type_node
            node = code_graph.get_out_neighbors_with_edge_type(node.id, FeatureEdge.NEXT_TOKEN)[0]
        return tokens

    def _get_bounds(self, method_block, code_graph):
        bounds = code_graph.get_neighbors_with_type_content(method_block.id,
                                                            neigh_type=FeatureNode.TOKEN,
                                                            neigh_content=None)
        start = bounds[0]
        end = bounds[1]
        if start.contents != LBRACE:
            t = start
            start = end
            end = t
        return start, end

    def _clean_tokens(self, tokens):
        return " ".join(self.text_filter.apply_to_token_lst(tokens))

    def _clean_javadoc(self, javadoc):
        javadoc_contents = javadoc.replace("/", "").replace("*", "") \
                                  .replace("{", "").replace("}", "").lower()
        cleaned_javadoc = self.text_filter.apply_to_javadoc(javadoc_contents)
        return " ".join(cleaned_javadoc)

    def _append_to_file(self, text, file_name):
        with open(file_name, "a") as file:
            file.write(text + "\n")

    def _split_camel_case(self, text):
        return re.sub('([a-z])([A-Z])', r'\1 \2', text).split()
