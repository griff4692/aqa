from collections import defaultdict
import os
import pickle
import re
import sys

from allennlp.predictors.predictor import Predictor
import matplotlib.pyplot as plt
import networkx as nx
import neuralcoref
from nlp import load_dataset
import spacy
from torch.utils.data import Subset

ARG_REGEX = r'\[ARG(\d{1,3}: [^\[\]]+)\]'
SIM_THRESHOLD = 0.7


def _node_match(new_node, graph, alt_names):
    match = None
    max_sim = 0.0
    for node in graph.nodes():
        full_comp = [node] + list(alt_names.get(new_node, []))
        max_overlap = max(list(map(lambda x: overlap(new_node, x), full_comp)))
        if max_overlap >= max(max_sim, SIM_THRESHOLD):
            print('Merging {} into {}'.format(new_node, node))
            max_sim = max_overlap
            match = node
    return match


def _sentence_split(str):
    return [s.text.strip() for s in sentence_nlp(str).sents]


def extract_args(description):
    arg_strs = re.findall(ARG_REGEX, description)
    results = []
    prev_d = -1
    for arg_str in arg_strs:
        d, arg = arg_str.split(':')
        d = int(d)
        if not d > prev_d:
            raise Exception(description)
        arg = arg.strip()
        prev_d = d
        results.append(arg.lower())
    return list(set(results))


def handle_node(node, graph, alt_names, node_weights):
    match = _node_match(node, graph, alt_names)
    resolved_node = match or node
    if match is None:
        graph.add_node(node, weight=1)
    else:
        alt_names[match].add(node)
    node_weights[resolved_node] += 1
    return resolved_node


def overlap(a, b):
    feats = tf_idf.transform([a.lower(), b.lower()])
    return (feats[0, :] * feats[1, :].T).todense()[0, 0]


def construct_graph(example, verbose=True):
    contexts = example['entity_pages']
    sents = []
    clusters = defaultdict(set)
    coref_contexts =  []
    for context in contexts['wiki_context']:
        coref = coref_nlp(context)._
        coref_contexts.append(coref.coref_resolved)
        sents += _sentence_split(coref.coref_resolved)
        cluster_set = coref.coref_clusters
        for cluster in cluster_set:
            mentions = list(map(lambda x: x.text.lower(), cluster.mentions))
            clusters[cluster.main.text.lower()].update(mentions)

    batch_sents = list(map(lambda x: {'sentence': x}, sents))
    ie_outputs = predictor.predict_batch_json(batch_sents)
    graph = nx.DiGraph()

    node_weights = defaultdict(int)
    alt_node_names = defaultdict(set)
    alt_edge_names = defaultdict(set)
    for ie_output in ie_outputs:
        for output in ie_output['verbs']:
            verb = output['verb'].lower()
            args = extract_args(output['description'])
            if len(args) < 2:
                continue
            subject = args[0]
            objects = args[1:]

            resolved_subject = handle_node(subject, graph, alt_node_names, node_weights)
            for obj in objects:
                resolved_obj = handle_node(obj, graph, alt_node_names, node_weights)
                if graph.has_edge(resolved_subject, resolved_obj):
                    prev_edge_data = graph.get_edge_data(resolved_subject, resolved_obj)
                    prev_weight, predicate = prev_edge_data['weight'], prev_edge_data['predicate']
                    graph.remove_edge(resolved_subject, resolved_obj)
                    graph.add_edge(resolved_subject, resolved_obj, weight=prev_weight + 1, predicate=predicate)
                    edge_str = '{}|{}|{}'.format(resolved_subject, resolved_obj, predicate)
                    alt_edge_names[edge_str].add(verb)
                else:
                    graph.add_edge(resolved_subject, resolved_obj, weight=1, predicate=verb)

    if verbose:
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos=pos)
        nx.draw_networkx_labels(graph, pos=pos)
        nx.draw_networkx_edge_labels(graph, pos=pos)
        plt.show()
    # Include question_id just to make sure we serialize in same order (as FK into full dataset)
    return {
        'question_id': example['question_id'],
        'alt_edge_names': alt_edge_names,
        'alt_node_names': alt_node_names,
        'coref_contexts': coref_contexts,
        'clusters': clusters,
        'graph': graph,
    }


if __name__ == '__main__':
    debug_mode = len(sys.argv) > 1 and sys.argv[1] == 'debug'

    print('Loading Dataset')
    dataset = load_dataset('trivia_qa', 'rc')
    print('Loading Coref Pipeline')
    coref_nlp = spacy.load('en_core_web_lg')
    coref = neuralcoref.NeuralCoref(coref_nlp.vocab)
    coref_nlp.add_pipe(coref, name='neuralcoref')

    sentence_nlp = spacy.load('en_core_web_lg')
    sentence_nlp.add_pipe(sentence_nlp.create_pipe('sentencizer'))

    predictor = Predictor.from_path(
        'https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz')

    with open('trivia_qa/tf_idf_vectorizer.pk', 'rb') as fd:
        tf_idf = pickle.load(fd)

    dtypes = list(dataset.keys())  # train, test, etc,
    for dtype in dtypes:
        d = dataset[dtype]
        if debug_mode:
            d = Subset(d, list(range(1, 3)))
        n = len(d)
        print('Processing {} examples from {} set'.format(n, dtype))
        out_fn = os.path.join('trivia_qa', 'graph_{}.pk'.format(dtype))

        graphs = list(map(construct_graph, d))
        qids = set(list(map(lambda c: c['question_id'], graphs)))
        print('Unique QIDs={}'.format(len(qids)))
        print(len(graphs))

        with open(out_fn, 'w') as fd:
            pickle.dump(graphs, fd)
        print('Saved pickled output to {}'.format(out_fn))
