import json
import pickle

SIM_THRESHOLD = 0.7


class Node:
    def __init__(self, name=None, weight=1):
        self.name = name
        self.weight = weight
        self.alts = set()

    def incr_weight(self):
        self.weight += 1

    def all_names(self):
        if self.name in self.alts:
            return self.alts
        return [self.name] + list(self.alts)

    def add_alt(self, alt_name):
        self.alts.add(alt_name)


def resolve_node_edge(new_node1_str, new_node2_str, edge_str, graph) -> Node:
    match1 = _node_match(new_node1_str, graph)
    match2 = _node_match(new_node2_str, graph)
    if new_node1_str != match1.name:
        match1.add_alt(new_node1_str)
        match1.incr_weight()

    if new_node2_str != match2.name:
        match2.add_alt(new_node2_str)
        match2.incr_weight()
       
    graph.add_node(match1.name)
    graph.add_node(match2.name)
    graph.add_edge(match1, match2, edge_str)
    

# def resolve_node(new_node_str, graph) -> Node:
#     match = _node_match(new_node_str, graph)
#     if new_node_str != match.name:
#         print('Merging \'{}\' --> \'{}\''.format(new_node_str, match.name))
#     graph.add_node(match)

def overlap(a, b):
    feats = tf_idf.transform([a.lower(), b.lower()])
    return (feats[0, :] * feats[1, :].T).todense()[0, 0]


class EdgeList:
    def __init__(self, u):
        self.u = u
        self.v = []
        self.edge_attrs = []

    def add_edge(self, v, attrs):
        self.v.append(v)
        self.edge_attrs.append(attrs)

    def incr_edge(self, edge_idx, predicate=None):
        self.edge_attrs[edge_idx]['weight'] += 1
        self.edge_attrs[edge_idx]['predicate'].add(predicate)

    def idx(self, v):
        for idx, node in enumerate(self.v):
            if node.name == v.name:
                return idx
        return None


class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_node(self, node_str):
        if node_str not in self.adj_list:
            u = Node(name=node_str, weight=1)
            self.adj_list[node_str] = EdgeList(u)

    def node_names(self):
        return list(self.adj_list.keys())

    def nodes(self):
        return [e.u for e in self.adj_list.values()]

    def add_edge(self, u, v, edge_attrs):
        if v not in self.adj_list[u.name].v and edge_attrs not in self.adj_list[u.name].edge_attrs:
            self.adj_list[u.name].add_edge(v, edge_attrs)

    def edge_idx(self, u, v):
        return self.adj_list[u].idx(v)


def _node_match(new_node_str, graph):
    match = Node(name=new_node_str, weight=1)
    max_sim = 0.0
    for node in graph.nodes():
        max_overlap = max(list(map(lambda x: overlap(new_node_str, x), node.all_names())))
        if max_overlap >= max(max_sim, SIM_THRESHOLD):
            max_sim = max_overlap
            match = node
            print('Merging \'{}\' --> \'{}\''.format(new_node_str, match.name))
    return match


if __name__ == '__main__':

    print('Loading tf-idf vectorizer...')
    with open('trivia_qa/tf_idf_vectorizer.pk', 'rb') as fd:
        tf_idf = pickle.load(fd)

    print('Loading context...')
    with open('trivia_qa/contexts_ie_debug.json', 'rb') as fd:
        contexts_ie_debug = json.load(fd)
        
    print('Loading data...')
    with open('trivia_qa/train_mini.json', 'rb') as fd:
        train_mini = json.load(fd)
        
    graph = Graph()
    
    for key in contexts_ie_debug.keys():
        for ie_tup in contexts_ie_debug[key]['ie_tups']:
            node_1, edge, node_2 = ie_tup
            resolve_node_edge(node_1, node_2, edge, graph)
        
