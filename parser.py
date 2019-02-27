#!/usr/bin/python

import os
import re

from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from text_filter import TextFilter
from code_graph import CodeGraph
from constants import *
from utils import append_to_file, remove_whitespace

class Parser:

    def __init__(self, tags_file, keywords_file, line_threshold=3):
        self.text_filter = TextFilter(tags_file, keywords_file)
        self.line_threshold = line_threshold
        self.camel_case_regex = re.compile('([a-z])([A-Z])')


    def generate_data_from_dir(self, base, output_folder="data"):
        num_methods_written = 0
        num_files_processed = 0
        for root, _dirs, files in os.walk(base):
            for file_name in files:
                file_path = root + "/" + file_name
                num_methods_written += self.generate_data_from_file(file_path, output_folder)
                num_files_processed += 1
                if (num_files_processed % 100 == 0):
                    print("Processed {0} files".format(num_files_processed))
        return num_methods_written

    def parse_directory(self, base):

        method_name_tokens = []
        api_call_tokens = []
        javadoc_tokens = []
        method_tokens = []
        method_bodies = []
        
        for root, _dirs, files in os.walk(base):
            for file_name in files:
                file_path = root + "/" + file_name
                tokens, api, name, javadoc, body = self.parse_file(file_path)

                method_name_tokens += tokens
                api_call_tokens += api
                javadoc_tokens += javadoc
                method_tokens += tokens
                method_bodies += body
                
        return method_tokens, api_call_tokens, method_name_tokens, javadoc_tokens, method_bodies

    def generate_data_from_file(self, file_name, output_folder="data"):
        method_name_file = output_folder + "/" + METHOD_NAME_FILE_NAME
        method_api_file = output_folder + "/" + METHOD_API_FILE_NAME
        method_tokens_file = output_folder + "/" + METHOD_TOKENS_FILE_NAME
        javadoc_tokens_file = output_folder + "/" + JAVADOC_FILE_NAME

        method_tokens, api_call_tokens, method_name_tokens, javadoc_tokens, _body = self.parse_file(file_name)

        if len(method_tokens) > 0 and len(api_call_tokens) > 0 and \
           len(method_name_tokens) > 0 and len(javadoc_tokens) > 0:
            append_to_file(method_tokens, method_tokens_file)
            append_to_file(api_call_tokens, method_api_file)
            append_to_file(method_name_tokens, method_name_file)
            append_to_file(javadoc_tokens, javadoc_tokens_file)
            return len(method_tokens)
        return 0

    def parse_file(self, file_name, only_javadoc=True):

        names = []
        apis = []
        javadocs = []
        tokens = []
        method_bodies = []

        with open(file_name, 'rb') as proto_file:
            g = Graph()
            try:
                g.ParseFromString(proto_file.read())
            except:
                print("Error parsing: " + file_name)
                return tokens, apis, names, javadocs, method_bodies

            code_graph = CodeGraph(g)

            method_dict = code_graph.methods if only_javadoc else code_graph.all_methods

            for method in method_dict.values():

                # We skip very small methods because these can often be described with
                # heuristics
                if method.num_lines <= self.line_threshold:
                    continue

                method_name_tokens = self._split_camel_case(method.method_name)
                method_name_tokens = [token.lower() for token in method_name_tokens]

                method_invocations = self._get_method_invocations(method.method_block, code_graph)
                api_call_tokens = []
                for invocation in method_invocations:
                    api_call_tokens.append(self._parse_method_invocation(invocation, code_graph))

                obj_init_tokens = self._get_object_inits(method.method_block, code_graph)
                api_call_tokens += obj_init_tokens
                api_call_tokens = remove_whitespace(api_call_tokens)

                javadoc_tokens = []

                # There may be no javadoc on methods which are used during testing
                if method.javadoc:
                    javadoc_tokens = self._clean_javadoc(method.javadoc.contents)

                method_tokens = self._get_method_tokens(method.method_block, code_graph)
                method_tokens = remove_whitespace(method_tokens)
                method_tokens = self._clean_tokens(method_tokens)

                method_str = self._method_to_str(method.method_block, code_graph)

                # During testing, we only omit methods for which there is no proper method body
                if not only_javadoc and len(method_str.strip()) > 0 and len(method_name_tokens) > 0:
                    names.append(" ".join(method_name_tokens))
                    apis.append(" ".join(api_call_tokens))
                    tokens.append(method_tokens)
                    javadocs.append(javadoc_tokens)
                    method_bodies.append(method_str)

                # DUring training, we only omit methods which have no name or javadoc description
                if only_javadoc and len(javadoc_tokens) > 0 and len(method_name_tokens) > 0:
                    names.append(" ".join(method_name_tokens))
                    apis.append(" ".join(api_call_tokens))
                    tokens.append(method_tokens)
                    javadocs.append(javadoc_tokens)
                    method_bodies.append(method_str)

        return tokens, apis, names, javadocs, method_bodies

    def _parse_method_invocation(self, method_invocation_node, code_graph):
        method_select = code_graph.get_neighbors_with_type_content(method_invocation_node.id,
                                                                   neigh_type=None,
                                                                   neigh_content=METHOD_SELECT)

        if len(method_select) == 0:
            return ""
        else:
            method_select = method_select[0]

        member_select = code_graph.get_neighbors_with_type_content(method_select.id,
                                                                   neigh_type=None,
                                                                   neigh_content=MEMBER_SELECT)
        if len(member_select) == 0:
            return ""
        else:
            member_select = member_select[0]
        
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
            if type_node != None:
                api_str = type_node.contents

        if len(api_str) == 0:
            return api_str
        else:
            return API_FORMAT.format(api_str, identifier.contents)

    def _find_variable_type(self, variable_node, code_graph):
        node = variable_node

        if node.contents == THIS:
            return code_graph.class_name_node

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
                                                                neigh_type=None,
                                                                neigh_content=STATEMENTS)

        if len(statements) == 0:
            return []
        statements = statements[0]

        bounds = code_graph.get_neighbors_with_type_content(method_block.id,
                                                            neigh_type=FeatureNode.TOKEN,
                                                            neigh_content=None)


        all_invocations = code_graph.get_nodes_with_content(METHOD_INVOCATION)
        startLine = method_block.startLineNumber
        endLine = method_block.endLineNumber

        method_invocations = list(filter(lambda m: m.startLineNumber >= startLine and \
                                                   m.endLineNumber <= endLine,
                                         all_invocations))

        top_level_invocations = []
        for m_invoc in method_invocations:
            is_top_level = True
            for m in method_invocations:
                if m.id == m_invoc.id:
                    continue

                if m_invoc.startPosition >= m.startPosition and \
                   m_invoc.endPosition <= m.endPosition:
                   is_top_level = False
                   break

            if is_top_level:
                top_level_invocations.append(m_invoc)
        return top_level_invocations

    def _get_method_tokens(self, method_block, code_graph):
        start, end = self._get_bounds(method_block, code_graph)

        tokens = []
        node = start
        while (node.id != end.id):
            # We only use identifier tokens in our method tokens
            if node.type == FeatureNode.IDENTIFIER_TOKEN:
                tokens += [token.lower() for token in self._split_camel_case(node.contents)]
            node = code_graph.get_out_neighbors_with_edge_type(node.id, FeatureEdge.NEXT_TOKEN)[0]
        return list(set(tokens))

    def _method_to_str(self, method_block, code_graph):
        _start, end = self._get_bounds(method_block, code_graph)
        method_str = ""

        body = code_graph.get_in_neighbors(method_block.id)
        if len(body) == 0:
            return method_str
        body = body[0]

        method = code_graph.get_in_neighbors(body.id)
        if len(method) == 0:
            return method_str
        method = method[0]

        modifiers = code_graph.get_neighbors_with_type_content(method.id,
                                                               neigh_type=None,
                                                               neigh_content=MODIFIERS)
        start = None
        if len(modifiers) == 0:
            # There are no modifiers on the method, so we start with the return type
            ret_type = code_graph.get_neighbors_with_type_content(method.id,
                                                                  neigh_type=None,
                                                                  neigh_content=RETURN_TYPE)
            if len(ret_type) == 0:
                return method_str

            # We traverse down to the return type
            start = ret_type[0]
            while start.type != FeatureNode.TOKEN:
                start = code_graph.get_out_neighbors(start.id)[0]

        else:
            modifiers = code_graph.get_neighbors_with_type_content(modifiers[0].id,
                                                                   neigh_type=None,
                                                                   neigh_content=MODIFIERS)
            if len(modifiers) == 0:
                return method_str

            # We omit the annotation AST nodes
            modifier_tokens = list(filter(lambda n: n.contents != ANNOTATIONS,
                                          code_graph.get_out_neighbors(modifiers[0].id)))

            if len(modifier_tokens) == 0:
                return method_str

            # We start with the first modifier token
            start = modifier_tokens[0]
            for i in range(1, len(modifier_tokens)):
                if start.startPosition > modifier_tokens[i].startPosition:
                    start = modifier_tokens[i]

        
        node = start
        while (node.id != end.id):
            contents = node.contents
            if node.type == FeatureNode.TOKEN:
                if contents in translate_dict:
                    contents = translate_dict[contents]
                contents = contents.lower()

            parents = code_graph.get_in_neighbors(node.id)

            is_string_literal = len(list(filter(lambda n: n.contents == STRING_LITERAL, parents))) > 0
            if is_string_literal:
                contents = STRING_FORMAT.format(contents)

            is_char_literal = len(list(filter(lambda n: n.contents == CHAR_LITERAL, parents))) > 0
            if is_char_literal:
                contents = CHAR_FORMAT.format(contents)

            method_str += TOKEN_FORMAT.format(contents)
            node = code_graph.get_out_neighbors_with_edge_type(node.id, FeatureEdge.NEXT_TOKEN)
            if node == None or len(node) == 0:
                break
            node = node[0]
        
        method_str += translate_dict[end.contents]
        return method_str


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

    def _split_camel_case(self, text):
        return self.camel_case_regex.sub(r'\1 \2', text).split()
