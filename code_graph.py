from graph_pb2 import Graph, FeatureNode, FeatureEdge

TYPE = "TYPE"
VARIABLE = "VARIABLE"


class CodeGraph:

    def __init__(self, ast_graph):

        self.nodes = { n.id : n for n in ast_graph.node }
        self.edges = { (e.sourceId, e.destinationId): e for e in ast_graph.edge }

        self.adj = {}
        for src in self.nodes.keys():
            self.adj[src] = []
            for dest in self.nodes.keys():
                if (src, dest) in self.edges:
                    self.adj[src].append(dest)

        self.vars = self._create_variables_dict()

    def get_nodes_with_type(self, node_type):
        return list(filter(lambda n: n.type == node_type, self.nodes.values()))

    def get_nodes_with_content(self, node_content):
        return list(filter(lambda n: n.contents == node_content, self.nodes.values()))

    def get_nodes_with_type_content(self, node_type, node_content):
        return list(filter(lambda n: n.contents == node_content and n.type == node_type, \
                           self.nodes.values()))

    def get_out_neighbors(self, node_id):
        return [self.nodes[i] for i in self.adj[node_id]]

    def get_neighbors_with_type_content(self, node_id, neigh_type, neigh_content):
        assert neigh_type != None or neigh_content != None

        neighbor_ids = self.adj[node_id]
        neighbors = []
        for node_id in neighbor_ids:
            node = self.nodes[node_id]
            if neigh_type == None and node.contents == neigh_content:
                neighbors.append(node)
            elif node.type == neigh_type and neigh_content == None:
                neighbors.append(node)
            elif node.type == neigh_type and node.contents == neigh_content:
                neighbors.append(node)
        return neighbors

    def get_predecessors(self, node_id):
        pred = []
        for n_id in self.adj:
            if node_id in self.adj[n_id]:
                pred.append(n_id)
        return pred

    def get_out_neighbors_with_edge_type(self, node_id, edge_type):
        out_edges = self.get_out_edges_with_type(node_id, edge_type)
        if len(out_edges) == 0:
            return None
        return [self.nodes[e.destinationId] for e in out_edges]

    def get_edges_with_type(self, edge_type):
        return list(filter(lambda id,e: e.type == edge_type, self.edges.values()))

    def get_out_edges(self, node_id):
        return list(map(lambda n_id: self.edges[(node_id, n_id)], self.adj[node_id]))

    def get_in_edges(self, node_id):
        return list(map(lambda n: self.edges[(n.id, node_id)], self.get_predecessors(node_id)))

    def get_out_edges_with_type(self, node_id, edge_type):
        return list(filter(lambda e: e.type == edge_type, self.get_out_edges(node_id)))

    def get_in_edges_with_type(self, node_id, edge_type):
        return list(filter(lambda e: e.type == edge_type, self.get_in_edges(node_id)))


    def _create_variables_dict(self):
        self.var_dict = {}
        var_ast_nodes = self.get_nodes_with_type_content(FeatureNode.AST_ELEMENT, VARIABLE)

        for var_ast_node in var_ast_nodes:
            type_node = self.get_neighbors_with_type_content(var_ast_node.id,
                                                              neigh_type=None,
                                                              neigh_content=TYPE)[0]
            var_type_node = self.get_out_neighbors_with_edge_type(type_node.id,
                                                                  FeatureEdge.ASSOCIATED_TOKEN)
            if var_type_node == None:
                continue
            var_type_node = var_type_node[0]

            var_node = self.get_out_neighbors(var_type_node.id)[0]
            self.var_dict[var_node.id] = Variable(var_type_node, var_node)

class Variable:

    def __init__(self, type_node, var_node):
        self.type_node = type_node
        self.var_node = var_node

class MethodBlock:

    def __init__(self, start_node, end_node):
        self.start_node = start_node
        self.end_node = end_node
