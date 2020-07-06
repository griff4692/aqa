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
import networkx as nx
import neuralcoref
from nlp import load_dataset
import spacy
from torch.utils.data import Subset

ARG_REGEX = r'\[ARG(\d{1,3}: [^\[\]]+)\]'
MODIFIER_REGEX = r'\[ARGM-[A-Z]+: [^\[\]]+)\]'
SIM_THRESHOLD = 0.7


def _node_match(new_node, graph, alt_names):
    match = None
    max_sim = 0.0
    for node in graph.nodes():
        full_comp = [node] + list(alt_names.get(new_node, []))
        max_overlap = max(list(map(lambda x: overlap(new_node, x), full_comp)))
        if max_overlap >= max(max_sim, SIM_THRESHOLD):
            print('Merging \'{}\' --> \'{}\''.format(new_node, node))
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


def process_imojie_output(token_ids):
    temp=" ".join(token_ids)
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


def construct_graph(example, verbose=True):
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

    graph = nx.DiGraph()

    node_weights = defaultdict(int)
    alt_node_names = defaultdict(set)
    alt_edge_names = defaultdict(set)

    for subject, verb, object in ie_tups:
        resolved_subject = handle_node(subject, graph, alt_node_names, node_weights)
        resolved_obj = handle_node(object, graph, alt_node_names, node_weights)
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


def process_dataset(d, dtype, debug_mode=False):
    debug_str = '_debug' if debug_mode else ''
    out_fn = os.path.join('trivia_qa', 'graph_{}{}.pk'.format(dtype, debug_str))

    graphs = list(map(lambda x: construct_graph(x, verbose=debug_mode), d))
    qids = set(list(map(lambda c: c['question_id'], graphs)))
    print('Unique QIDs={}'.format(len(qids)))
    print(len(graphs))

    with open(out_fn, 'w') as fd:
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
