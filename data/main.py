from collections import defaultdict
import os
import json
import pickle
import re
import sys

from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules, sanitize
import_submodules('imojie')
import matplotlib.pyplot as plt
import neuralcoref
from nlp import load_dataset
import spacy
from torch.utils.data import Subset

ARG_REGEX = r'\[ARG(\d{1,3}: [^\[\]]+)\]'
MODIFIER_REGEX = r'\[ARGM-[A-Z]+: [^\[\]]+)\]'
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
        assert node_str not in self.adj_list
        u = Node(name=node_str, weight=1)
        self.adj_list[node_str] = EdgeList(u)

    def node_names(self):
        return list(self.adj_list.keys())

    def nodes(self):
        return [e.u for e in self.adj_list.values()]

    def add_edge(self, u, v, edge_attrs):
        self.adj_list[u.name].add_edge(v, edge_attrs)

    def edge_idx(self, u, v):
        return self.adj_list[u].idx(v)


def _node_match(new_node_str, graph):
    match = None
    max_sim = 0.0
    for node in graph.nodes():
        max_overlap = max(list(map(lambda x: overlap(new_node_str, x), node.all_names())))
        if max_overlap >= max(max_sim, SIM_THRESHOLD):
            max_sim = max_overlap
            match = node
    return match


def _sentence_split(str):
    return [s.text.strip() for s in sentence_nlp(str).sents]


def extract_args(description):
    arg_strs = re.findall(ARG_REGEX, description)
    results = defaultdict(list)
    modifiers = {}
    for arg_str in arg_strs:
        d, arg = arg_str.split(':')
        d = int(d)
        arg = arg.strip()
        results[d].append(arg.lower())

    modifier_strs = re.findall(MODIFIER_REGEX, description)
    for mod_str in modifier_strs:
        mod, text = mod_str.split(':')
        modifiers[mod.strip()] = text.strip().lower()
    return results, modifiers


def resolve_node(new_node_str, graph) -> Node:
    match = _node_match(new_node_str, graph)
    if match is None:
        match = graph.add_node(new_node_str)
    else:
        print('Merging \'{}\' --> \'{}\''.format(new_node_str, match.name))
        match.add_alt(new_node_str)
        match.incr_weight()
    return match


def overlap(a, b):
    feats = tf_idf.transform([a.lower(), b.lower()])
    return (feats[0, :] * feats[1, :].T).todense()[0, 0]


def process_imojie_output(token_ids):
    temp = ' '.join(token_ids)
    temp = temp.replace(" ##","")
    temp = temp.replace("[unused1]","( ")
    temp = temp.replace("[unused2]"," ; ")
    temp = temp.replace("[unused3]","")
    temp = temp.replace("[unused4]"," ; ")
    temp = temp.replace("[unused5]","")
    temp = temp.replace("[unused6]"," )")
    temp = temp.strip()
    temp = temp.split('[SEP]')
    ans = []
    for x in temp:
        if x != '':
            ans.append(x)
    return ans


def extract_tups_from_str(str):
    return list(map(lambda x: x.strip().lower(), str.strip('() ').split(';')))


def _construct_ie_tuples(args, idx):
    if idx not in args:
        return [[]]

    results = []
    next_paths = _construct_ie_tuples(args, idx + 1)
    for arg in args[idx]:
        for path in next_paths:
            results.append([arg] + path)
    return results


def construct_ie_tuples(args, verb):
    all_paths = _construct_ie_tuples(args, 0)
    all_paths_w_verb = map(lambda x: x[0] + [verb] + x[1:], all_paths)
    return all_paths_w_verb


def construct_graph(example):
    contexts = example['entity_pages']
    sents = []
    clusters = defaultdict(set)
    coref_contexts = []
    for context in contexts['wiki_context']:
        coref = coref_nlp(context)._
        coref_contexts.append(coref.coref_resolved)
        sents += _sentence_split(coref.coref_resolved)
        cluster_set = coref.coref_clusters
        for cluster in cluster_set:
            mentions = list(map(lambda x: x.text.lower(), cluster.mentions))
            clusters[cluster.main.text.lower()].update(mentions)

    ie_tups = []
    for sent in sents:
        inp_instance = predictor._dataset_reader.text_to_instance(sent)
        output = predictor._model.forward_on_instance(inp_instance)
        output = sanitize(output)
        output = process_imojie_output(output["predicted_tokens"][0])
        ie_tup = list(map(extract_tups_from_str, output))
        ie_tup_valid = list(filter(lambda x: len(x) == 3, ie_tup))

        if len(ie_tup_valid) < len(ie_tup):
            invalid_tups = list(set(ie_tup) - set(ie_tup_valid))
            print('Invalid tuples:\n\t{}'.format('\n'.join(invalid_tups)))
        ie_tups += ie_tup_valid

    graph = Graph()
    for subject, verb, object in ie_tups:
        u = resolve_node(subject, graph)
        v = resolve_node(object, graph)
        edge_idx = graph.edge_idx(u, v)
        if edge_idx is not None:
            graph.adj_list[u.name].incr_edge(edge_idx, predicate=verb)
        else:
            attrs = {
                'weight': 1,
                'predicates': set([verb])
            }
            graph.add_edge(u, v, attrs)

    # Include question_id just to make sure we serialize in same order (as FK into full dataset)
    obj_update = {
        'question_id': example['question_id'],
        'coref_contexts': coref_contexts,
        'clusters': clusters,
        'graph': graph,
    }

    example.update(obj_update)
    return example


def process_dataset(d, dtype, debug_mode=False):
    debug_str = '_debug' if debug_mode else ''
    out_fn = os.path.join('trivia_qa', 'graph_{}{}.pk'.format(dtype, debug_str))

    graphs = list(map(construct_graph, d))
    qids = set(list(map(lambda c: c['question_id'], graphs)))
    print('Unique QIDs={}'.format(len(qids)))
    print(len(graphs))

    with open(out_fn, 'wb') as fd:
        pickle.dump(graphs, fd)
    print('Saved pickled output to {}'.format(out_fn))


if __name__ == '__main__':
    debug_mode = len(sys.argv) > 1 and sys.argv[1] == 'debug'

    print('Loading Dataset')
    debug_data_fn = 'trivia_qa/train_mini.json'
    if debug_mode and os.path.exists(debug_data_fn):
        with open(debug_data_fn, 'r') as fd:
            dataset = json.load(fd)
    elif debug_mode and not os.path.exists(debug_data_fn):
        dataset = load_dataset('trivia_qa', 'rc')['train']
        dataset = [dict(e) for e in Subset(dataset, list(range(1, 11)))]
        with open(debug_data_fn, 'w') as fd:
            json.dump(dataset, fd)
    else:
        dataset = load_dataset('trivia_qa', 'rc')
    print('Loading Coref Pipeline')
    coref_nlp = spacy.load('en_core_web_lg')
    coref = neuralcoref.NeuralCoref(coref_nlp.vocab)
    coref_nlp.add_pipe(coref, name='neuralcoref')

    sentence_nlp = spacy.load('en_core_web_lg')
    sentence_nlp.add_pipe(sentence_nlp.create_pipe('sentencizer'))

    archive = load_archive(
        "../models/imojie",
        weights_file="../models/imojie/model_state_epoch_7.th",
        cuda_device=-1)

    predictor = Predictor.from_archive(archive, "noie_seq2seq")

    with open('trivia_qa/tf_idf_vectorizer.pk', 'rb') as fd:
        tf_idf = pickle.load(fd)

    if debug_mode:
        print('Processing {} size sample train examples in debug mode'.format(len(dataset)))
        process_dataset(dataset, 'train', debug_mode=True)
    else:
        dtypes = list(dataset.keys())  # train, test, etc
        for dtype in dtypes:
            d = dataset[dtype]
            n = len(d)
            print('Processing {} examples from {} set'.format(n, dtype))
            process_dataset(d, dtype, debug_mode=True)
