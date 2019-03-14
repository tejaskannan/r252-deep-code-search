from graph_pb2 import Graph, FeatureNode, FeatureEdge
from constants import *


class CodeGraph:
    """Class which holds an AST and provides a set of utility functions to access the graph."""

    def __init__(self, ast_graph):

        self.nodes = {n.id: n for n in ast_graph.node}
        self.edges = {(e.sourceId, e.destinationId): e for e in ast_graph.edge}

        # Forward and backward adjacency lists
        self.adj = {n.id: [] for n in ast_graph.node}
        self.rev_adj = {n.id: [] for n in ast_graph.node}

        for src, dest in self.edges.keys():
            self.adj[src].append(dest)
            self.rev_adj[dest].append(src)

        # Stores methods with javadoc annotations (used during training and validation)
        self.methods = self._create_javadoc_method_dict()

        # Dictionary of variables to their types
        self.vars = self._create_variables_dict()

        # Stores all methods in a dictionary (used during testing)
        self.all_methods = self._create_all_methods_dict()

        # AST Node which holds the class name
        self.class_name_node = self._get_class_name_node()

    def get_nodes_with_type(self, node_type):
        """Returns all nodes with the given type"""
        return list(filter(lambda n: n.type == node_type, self.nodes.values()))

    def get_nodes_with_content(self, node_content):
        """Returns all nodes with the given contents"""
        return list(filter(lambda n: n.contents == node_content, self.nodes.values()))

    def get_nodes_with_type_content(self, node_type, node_content):
        """Returns all nodes with both the given type and the given content"""
        return list(filter(lambda n: n.contents == node_content and n.type == node_type,
                           self.nodes.values()))

    def get_out_neighbors(self, node_id):
        """Returns all outgoing neighbors."""
        return [self.nodes[i] for i in self.adj[node_id]]

    def get_in_neighbors(self, node_id):
        """Returns all incoming neighbors."""
        return [self.nodes[i] for i in self.rev_adj[node_id]]

    def get_neighbors_with_type_content(self, node_id, neigh_type, neigh_content):
        """
        Returns all outgoing neighboring nodes withe the given type and content.
        If the type or content is None, the property is ignored.
        """
        assert neigh_type is not None or neigh_content is not None, 'Must have either Type or Content'

        neighbor_ids = self.adj[node_id]
        neighbors = []
        for node_id in neighbor_ids:
            node = self.nodes[node_id]
            if neigh_type is None and node.contents == neigh_content:
                neighbors.append(node)
            elif node.type == neigh_type and neigh_content is None:
                neighbors.append(node)
            elif node.type == neigh_type and node.contents == neigh_content:
                neighbors.append(node)
        return neighbors

    def get_predecessors(self, node_id):
        """Returns the ids of all incoming neighbors."""
        return self.rev_adj[node_id]

    def get_out_neighbors_with_edge_type(self, node_id, edge_type):
        """
        Returns all outgoing neighbors which are connected to the given node
        with edges of the given edge type.
        """
        out_edges = self.get_out_edges_with_type(node_id, edge_type)
        if len(out_edges) == 0:
            return None
        return [self.nodes[e.destinationId] for e in out_edges]

    def get_in_neighbors_with_edge_type(self, node_id, edge_type):
        """
        Returns all incoming neighbors which are connected to the given node
        with edges of the given edge type.
        """
        in_edges = self.get_in_edges_with_type(node_id, edge_type)
        if len(in_edges) == 0:
            return None
        return [self.nodes[e.sourceId] for e in in_edges]

    def get_edges_with_type(self, edge_type):
        """Returns all edges with the given type."""
        return list(filter(lambda e: e.type == edge_type, self.edges.values()))

    def get_out_edges(self, node_id):
        """Returns all outgoing edges of the given node."""
        return list(map(lambda n_id: self.edges[(node_id, n_id)], self.adj[node_id]))

    def get_in_edges(self, node_id):
        """Returns all incoming edges to the given node."""
        return list(map(lambda n: self.edges[(n, node_id)], self.get_predecessors(node_id)))

    def get_out_edges_with_type(self, node_id, edge_type):
        """Returns all outgoing edges of the given node with the given edge type."""
        return list(filter(lambda e: e.type == edge_type, self.get_out_edges(node_id)))

    def get_in_edges_with_type(self, node_id, edge_type):
        """Returns all incoming edges to the given node with the given edge type."""
        return list(filter(lambda e: e.type == edge_type, self.get_in_edges(node_id)))

    def _create_variables_dict(self):
        """
        Returns a dictionary which maps node IDs to a Variable object which holds the variable
        node and its type node.
        """
        var_dict = {}

        # Fetch all Variable AST nodes
        var_ast_nodes = self.get_nodes_with_type_content(FeatureNode.AST_ELEMENT, VARIABLE)

        for var_ast_node in var_ast_nodes:

            # Find the variable's associated type node
            type_node = self.get_neighbors_with_type_content(var_ast_node.id,
                                                             neigh_type=None,
                                                             neigh_content=TYPE)
            if type_node is None or len(type_node) == 0:
                continue
            else:
                type_node = type_node[0]

            type_node = self.get_out_neighbors(type_node.id)[0]
            if type_node.type != FeatureNode.IDENTIFIER_TOKEN:
                # If the type is not an identifier token, the it may be either parameterized or primitive
                if type_node.contents == PARAM_TYPE:
                    type_node = self.get_neighbors_with_type_content(type_node.id,
                                                                     neigh_type=None,
                                                                     neigh_content=TYPE)[0]
                type_node = self.get_out_neighbors(type_node.id)[0]

            var_node = self.get_neighbors_with_type_content(var_ast_node.id,
                                                            neigh_type=FeatureNode.IDENTIFIER_TOKEN,
                                                            neigh_content=None)[0]
            var_dict[var_node.id] = Variable(type_node, var_node)
        return var_dict

    def _create_javadoc_method_dict(self):
        """
        Returns a dictionary which maps a method block ids to Method objects for only methods which
        have an associated Javadoc comment.
        """
        method_dict = {}
        javadoc_nodes = self.get_nodes_with_type(FeatureNode.COMMENT_JAVADOC)
        for javadoc_node in javadoc_nodes:
            method_nodes = self.get_neighbors_with_type_content(javadoc_node.id,
                                                                neigh_type=None,
                                                                neigh_content=METHOD)
            if len(method_nodes) == 0:
                continue

            method_node = method_nodes[0]
            self._add_method(method_node, javadoc_node, method_dict)

        return method_dict

    def _create_all_methods_dict(self):
        """
        Returns a dictionary which maps a method block ids to Method objects for all methods
        even if they don't have an associated Javadoc comment.
        """
        method_dict = {}
        method_nodes = self.get_nodes_with_content(METHOD)

        # First fetch all javadoc comments and store in a dictionary. This information
        # is used to include these nodes in the later Method objects.
        javadoc_dict = {}
        javadoc_nodes = self.get_nodes_with_type(FeatureNode.COMMENT_JAVADOC)
        for javadoc_node in javadoc_nodes:
            method = self.get_neighbors_with_type_content(javadoc_node.id, None, METHOD)
            if len(method) == 0:
                continue
            method = method[0]
            javadoc_dict[method.id] = javadoc_node

        # Add each method to the dictionary of methods
        for method_node in method_nodes:
            javadoc = javadoc_dict[method_node.id] if method_node.id in javadoc_dict else None
            self._add_method(method_node, javadoc, method_dict)
        return method_dict

    def _add_method(self, method_node, javadoc_node, method_dict):
        """
        Adds the given method to the given method dictionary.
        Includes the given javadoc node.
        """

        # Traverse down to the method_block node
        method_assoc_tokens = self.get_out_neighbors_with_edge_type(method_node.id,
                                                                    edge_type=FeatureEdge.ASSOCIATED_TOKEN)
        if method_assoc_tokens is None or len(method_assoc_tokens) == 0:
            return

        method_name_node = next(filter(lambda n: n.type == FeatureNode.IDENTIFIER_TOKEN,
                                       method_assoc_tokens))

        method_body = self.get_neighbors_with_type_content(method_node.id,
                                                           neigh_type=None,
                                                           neigh_content=BODY)

        if len(method_body) == 0:
            return

        method_body = method_body[0]
        method_block = self.get_neighbors_with_type_content(method_body.id,
                                                            neigh_type=None,
                                                            neigh_content=BLOCK)

        if len(method_block) == 0:
            return

        # Use the method block node to create the Method object
        method_block = method_block[0]
        num_lines = method_node.endLineNumber - method_node.startLineNumber
        method_dict[method_block.id] = Method(javadoc_node, method_block,
                                              method_name_node.contents,
                                              num_lines)

    def _get_class_name_node(self):
        """
        Returns the AST node associated with the class name.
        """
        compilation_unit = self.get_nodes_with_content(COMPILATION_UNIT)
        if len(compilation_unit) == 0:
            return None
        compilation_unit = compilation_unit[0]

        comp_unit_neighbors = self.get_out_neighbors(compilation_unit.id)
        class_name_node = list(filter(lambda n: n.type == FeatureNode.IDENTIFIER_TOKEN, comp_unit_neighbors))
        if len(class_name_node) == 0:
            return None
        return class_name_node[0]


class Variable:
    """Wrapper class for variables."""

    def __init__(self, type_node, var_node):
        self.type_node = type_node
        self.var_node = var_node


class Method:
    """Wrapper class for methods."""

    def __init__(self, javadoc, method_block, method_name, num_lines):
        self.javadoc = javadoc
        self.method_block = method_block
        self.method_name = method_name
        self.num_lines = num_lines
