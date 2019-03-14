import os
import re
from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from text_filter import TextFilter
from code_graph import CodeGraph
from constants import *
from utils import append_to_file, remove_whitespace


class Parser:
    """Class used to parse Java ASTs which are serialized as protobuf files."""

    def __init__(self, tags_file, stopwords_file, line_threshold=2):
        self.text_filter = TextFilter(tags_file, stopwords_file)
        self.line_threshold = line_threshold
        self.hex_regex = re.compile(r'[a-f]+')

    def generate_data_from_dir(self, base, output_folder, should_subtokenize=False):
        """
        Extracts features from all methods in the given base folder and writes outputs
        to the files method-names.txt, method-apis.txt, method-tokens.txt, and javadoc.txt located
        in the output folder.
        """
        num_methods_written = 0
        num_files_processed = 0
        for root, _dirs, files in os.walk(base):
            for file_name in files:
                file_path = root + '/' + file_name
                num_methods_written += self.generate_data_from_file(file_path, output_folder,
                                                                    should_subtokenize=should_subtokenize)
                num_files_processed += 1
                if (num_files_processed % REPORT_THRESHOLD == 0):
                    print('Processed {0} files'.format(num_files_processed))
        return num_methods_written

    def parse_directory(self, base, should_subtokenize=False):
        """
        Returns features extracted from the protobuf files located in the given
        directory.
        """

        method_name_tokens = []
        api_call_tokens = []
        javadoc_tokens = []
        method_tokens = []
        method_bodies = []

        for root, _dirs, files in os.walk(base):
            for file_name in files:
                file_path = root + '/' + file_name
                tokens, api, name, javadoc, body = self.parse_file(file_path,
                                                                   should_subtokenize=should_subtokenize)

                method_name_tokens += tokens
                api_call_tokens += api
                javadoc_tokens += javadoc
                method_tokens += tokens
                method_bodies += body

        return method_tokens, api_call_tokens, method_name_tokens, javadoc_tokens, method_bodies

    def generate_data_from_file(self, file_name, output_folder, should_subtokenize=False):
        """
        Appends the features extracted from methods located in the given file to the output files
        method-names.txt, method-apis.txt, method-tokens.txt and javadoc.txt. These files are located
        in the given output folder. This function returns the number of methods written.
        """
        name_file = output_folder + '/' + METHOD_NAME_FILE_NAME
        api_file = output_folder + '/' + METHOD_API_FILE_NAME
        tokens_file = output_folder + '/' + METHOD_TOKENS_FILE_NAME
        javadoc_file = output_folder + '/' + JAVADOC_FILE_NAME

        tokens, api_calls, names, javadocs, _body = self.parse_file(file_name,
                                                                    should_subtokenize=should_subtokenize)

        if len(tokens) > 0 and len(api_calls) > 0 and len(names) > 0 and len(javadocs) > 0:
            append_to_file(tokens, tokens_file)
            append_to_file(api_calls, api_file)
            append_to_file(names, name_file)
            append_to_file(javadocs, javadoc_file)
            return len(tokens)
        return 0

    def parse_file(self, file_name, only_javadoc=True, lowercase_api=True, should_subtokenize=False):
        """
        Extracts features from a single protobuf file.
        """

        names = []
        apis = []
        javadocs = []
        tokens = []
        method_bodies = []

        with open(file_name, 'rb') as proto_file:
            g = Graph()

            # Parse protobuf file as a graph. Skips all files which error.
            try:
                g.ParseFromString(proto_file.read())
            except:
                print('Error parsing: {0}'.format(file_name))
                return tokens, apis, names, javadocs, method_bodies

            code_graph = CodeGraph(g)

            # We either extract features from all method or those which have associated
            # Javadoc comments.
            method_dict = code_graph.methods if only_javadoc else code_graph.all_methods

            for method in method_dict.values():

                # Omit methods which are below the defined threshold
                if method.num_lines <= self.line_threshold:
                    continue

                # Parse method name tokens
                method_name_tokens = self.text_filter.apply_to_method_name(method.method_name)

                # Parse API invocations
                method_invocations = self._get_method_invocations(method.method_block, code_graph)
                api_call_tokens = []
                for invocation in method_invocations:
                    parsed_invocation = self._parse_method_invocation(invocation, code_graph).strip()
                    api_call_tokens.append(parsed_invocation)

                obj_init_tokens = self._get_object_inits(method.method_block, code_graph)
                api_call_tokens += obj_init_tokens
                api_call_tokens = self.text_filter.apply_to_api_calls(api_call_tokens, lowercase_api,
                                                                      should_subtokenize=should_subtokenize)

                # Parse Javadoc comments. We check to make sure the method has an associdated
                # Javadoc comment, as there may be no javadoc on methods which are used during testing.
                javadoc_tokens = []
                if method.javadoc:
                    javadoc_tokens = self.text_filter.apply_to_javadoc(method.javadoc.contents)

                # Parse method tokens
                method_tokens = self._get_method_tokens(method.method_block, code_graph)
                method_tokens = self.text_filter.apply_to_token_lst(method_tokens)

                # Extract the entire method body. This field is used during searching.
                method_str = self._method_to_str(method.method_block, code_graph)

                # During testing, we only omit methods for which there is no proper method body
                if not only_javadoc and len(method_str.strip()) > 0 and len(method_name_tokens) > 0:
                    # Tokens in the output files are separated by spaces
                    names.append(' '.join(method_name_tokens))
                    apis.append(' '.join(api_call_tokens))
                    tokens.append(' '.join(method_tokens))
                    javadocs.append(' '.join(javadoc_tokens))
                    method_bodies.append(method_str)

                # During training, we only omit methods which have no name or javadoc description
                if only_javadoc and len(javadoc_tokens) > 0 and len(method_name_tokens) > 0:
                    # Tokens in the output files are separated by spaces
                    names.append(' '.join(method_name_tokens))
                    apis.append(' '.join(api_call_tokens))
                    tokens.append(' '.join(method_tokens))
                    javadocs.append(' '.join(javadoc_tokens))
                    method_bodies.append(method_str)

        return tokens, apis, names, javadocs, method_bodies

    def _parse_method_invocation(self, method_invocation_node, code_graph):
        """
        Returns a string which represents a single API invocation.
        """

        # Walks the graph until the identifier node for this method call is found.
        method_select = code_graph.get_neighbors_with_type_content(method_invocation_node.id,
                                                                   neigh_type=None,
                                                                   neigh_content=METHOD_SELECT)
        if len(method_select) == 0:
            return ''
        else:
            method_select = method_select[0]

        member_select = code_graph.get_neighbors_with_type_content(method_select.id,
                                                                   neigh_type=None,
                                                                   neigh_content=MEMBER_SELECT)
        if len(member_select) == 0:
            return ''
        else:
            member_select = member_select[0]

        expr_node = code_graph.get_neighbors_with_type_content(member_select.id,
                                                               neigh_type=FeatureNode.FAKE_AST,
                                                               neigh_content=EXPRESSION)[0]
        identifier = code_graph.get_neighbors_with_type_content(member_select.id,
                                                                neigh_type=FeatureNode.IDENTIFIER_TOKEN,
                                                                neigh_content=None)[0]

        # Fetch the token node. If no token node exists, then there is a nested API call.
        token_node = code_graph.get_out_neighbors_with_edge_type(expr_node.id, FeatureEdge.ASSOCIATED_TOKEN)

        api_str = ''
        if token_node is None:
            # This means we have a nested method call. We recursively call this function to
            # search the graph in a depth-first manner.
            inner_invocation = code_graph.get_out_neighbors_with_edge_type(expr_node.id, FeatureEdge.AST_CHILD)[0]
            api_str = self._parse_method_invocation(inner_invocation, code_graph)
        else:
            # Extract the token and if it is a variable name, replace it with its type
            token = token_node[0]
            type_node = self._find_variable_type(token, code_graph)
            if type_node is not None:
                api_str = type_node.contents

        if len(api_str) == 0:
            return api_str

        return API_FORMAT.format(api_str, identifier.contents)

    def _find_variable_type(self, variable_node, code_graph):
        """Returns the AST Node which represents the given variable's type."""
        node = variable_node

        if node.contents == THIS:
            return code_graph.class_name_node

        # Move forward along the last lexical use edges
        while node:
            # If this node contains the declaration of this variable, then
            # return the type node.
            if node.id in code_graph.vars:
                variable = code_graph.vars[node.id]
                return variable.type_node
            next_use = code_graph.get_out_neighbors_with_edge_type(node.id,
                                                                   FeatureEdge.LAST_LEXICAL_USE)
            if next_use:
                node = next_use[0]
            else:
                node = next_use

        # Move backward along the last lexical use edges
        node = variable_node
        while node:
            # If this node contains the declaration of this variable, then
            # return the type node.
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
        """
        Returns all top-level API invocations in the given method. All nested calls are separately
        handled when parsing API calls.
        """
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

        # Fetch all API invocations located within this method
        method_invocations = list(filter(lambda m: m.startLineNumber >= startLine and \
                                         m.endLineNumber <= endLine,
                                         all_invocations))

        # Find all top-level invocations using start and end position bounds.
        top_level_invocations = []
        for m_invoc in method_invocations:
            is_top_level = True
            for m in method_invocations:
                if m.id == m_invoc.id:
                    continue

                # An invocation is not top-level if there exists a method call block which
                # entirely encompasseses the current block.
                if m_invoc.startPosition >= m.startPosition and \
                   m_invoc.endPosition <= m.endPosition:
                    is_top_level = False
                    break

            if is_top_level:
                top_level_invocations.append(m_invoc)
        return top_level_invocations

    def _get_method_tokens(self, method_block, code_graph):
        """Returns a list of unique, non-keyword tokens found in the given method."""
        start, end = self._get_bounds(method_block, code_graph)

        tokens = []
        node = start
        while (node.id != end.id):
            # We only use identifier tokens as all keywords and operations are omitted
            if node.type == FeatureNode.IDENTIFIER_TOKEN:
                tokens.append(node.contents)
            node = code_graph.get_out_neighbors_with_edge_type(node.id, FeatureEdge.NEXT_TOKEN)[0]

        # Return only unique tokens
        return list(set(tokens))

    def _method_to_str(self, method_block, code_graph):
        """
        Returns the given method as a single string.
        """
        _start, end = self._get_bounds(method_block, code_graph)
        method_str = ''

        body = code_graph.get_in_neighbors(method_block.id)
        if len(body) == 0:
            return method_str
        body = body[0]

        method = code_graph.get_in_neighbors(body.id)
        if len(method) == 0:
            return method_str
        method = method[0]

        # Fetch the nodes which denote the modifiers of this method
        modifiers = code_graph.get_neighbors_with_type_content(method.id,
                                                               neigh_type=None,
                                                               neigh_content=MODIFIERS)

        # Find the starting token for this method
        start = None
        if len(modifiers) == 0:
            # There are no modifiers on the method, so we start with the return type
            ret_type = code_graph.get_neighbors_with_type_content(method.id,
                                                                  neigh_type=None,
                                                                  neigh_content=RETURN_TYPE)

            if len(ret_type) == 0:
                return method_str

            # Traverse down to the return type.
            start = ret_type[0]
            while start.type not in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN):
                out_neighbors = code_graph.get_out_neighbors(start.id)
                start = out_neighbors[0]
                for i in range(1, len(out_neighbors)):

                    # Find the earliest token. This step is necessary to properly parse
                    # parameterized return types.
                    if start.startPosition > out_neighbors[i].startPosition:
                        start = out_neighbors[i]
        else:
            modifiers = code_graph.get_neighbors_with_type_content(modifiers[0].id,
                                                                   neigh_type=None,
                                                                   neigh_content=MODIFIERS)
            if len(modifiers) == 0:
                return method_str

            modifier_tokens = list(filter(lambda n: n.contents != ANNOTATIONS,
                                          code_graph.get_out_neighbors(modifiers[0].id)))

            if len(modifier_tokens) == 0:
                return method_str

            # Start with the first modifier token as methods can have many modifiers (i.e. public static)
            start = modifier_tokens[0]
            for i in range(1, len(modifier_tokens)):
                if start.startPosition > modifier_tokens[i].startPosition:
                    start = modifier_tokens[i]

        # Iterate through all tokens in this method
        node = start
        while (node.id != end.id):
            contents = node.contents

            # Translate regular tokens into more human-readable formats (i.e. EQ becomes =)
            if node.type == FeatureNode.TOKEN:
                if node.contents in translate_dict:
                    contents = translate_dict[node.contents]
                else:
                    contents = ''
                    for i, c in enumerate(node.contents):
                        if c in translate_dict:
                            c = translate_dict[c]
                        contents += c
                contents = contents.lower()

            parents = code_graph.get_in_neighbors(node.id)

            # Format string and character literals
            is_string_literal = len(list(filter(lambda n: n.contents == STRING_LITERAL, parents))) > 0
            if is_string_literal:
                contents = STRING_FORMAT.format(contents)

            is_char_literal = len(list(filter(lambda n: n.contents == CHAR_LITERAL, parents))) > 0
            if is_char_literal:
                contents = CHAR_FORMAT.format(contents)

            # Addresses the issue of displaying hexidecimal numbers
            is_int_literal = len(list(filter(lambda n: n.contents == INT_LITERAL, parents))) > 0
            if is_int_literal:
                contents = contents.lower()
                is_hex = self.hex_regex.search(contents)
                if is_hex:
                    contents = '0x' + contents

            # Concatenate this token
            method_str += TOKEN_FORMAT.format(contents)
            node = code_graph.get_out_neighbors_with_edge_type(node.id, FeatureEdge.NEXT_TOKEN)
            if node is None or len(node) == 0:
                break
            node = node[0]

        # Translate the last token from RBRACE to }
        method_str += translate_dict[end.contents]
        return method_str

    def _get_object_inits(self, method_block, code_graph):
        """
        Returns a list of strings representating of new object intiializations.
        Type parameters are omitted. For example, 'new ArrayList<Integer>'' is parsed as 'arraylist.new'
        """
        start, end = self._get_bounds(method_block, code_graph)

        tokens = []
        node = start
        while (node.id != end.id):
            # Parse nodes which contain the 'new' keyword
            if node.contents == NEW:
                obj_type_node = code_graph.get_out_neighbors_with_edge_type(node.id, FeatureEdge.NEXT_TOKEN)[0]
                tokens.append(API_FORMAT.format(obj_type_node.contents, NEW_LOWER))
                node = obj_type_node
            node = code_graph.get_out_neighbors_with_edge_type(node.id, FeatureEdge.NEXT_TOKEN)[0]
        return tokens

    def _get_bounds(self, method_block, code_graph):
        """Returns the start and end nodes for the given method block."""
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
