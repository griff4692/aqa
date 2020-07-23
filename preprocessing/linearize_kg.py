from collections import Counter
import os
import pickle
from string import punctuation
import sys
import unicodedata

import argparse
import spacy
from torch.utils.data import Dataset

from dataset_base import dataset_factory
from kg import tokenize

import networkx as nx
from networkx.algorithms.traversal.breadth_first_search import bfs_edges

print('Loading Spacy...')
spacy_nlp = spacy.load('en_core_web_lg')
print('Finished loading Spacy...')


def tokenize(str):
    toks = [token.text.lower().strip(punctuation).strip() for token in spacy_nlp(str)]
    return [tok for tok in toks if len(tok) > 0]


def overlap(a, b):
    avg_len = float(len(a) + len(b)) / 2.0
    c = float(len(a.intersection(b)))
    return c / avg_len


def linearize_graph(graph, answer_toks, format='t5'):
    answer_tok_set = set(answer_toks)
    node_objects = list(graph.nodes.data())
    target_node_idx, max_overlap = None, 0.0
    for node in node_objects:
        node_idx, metadata = node
        head_tok_set = set(metadata['head'].split(' '))
        o = overlap(answer_tok_set, head_tok_set)
        if o > max_overlap:
            target_node_idx = node_idx
            max_overlap = o

    if target_node_idx is None:
        return '', 0.0, -1

    output = []
    for i, edge in enumerate(bfs_edges(graph, target_node_idx)):
        edge_data = graph.edges[edge]
        verbs = edge_data['verbs']
        if i > 0:
            verbs -= set(['subset'])
        verbs = list(verbs)
        if len(verbs) == 0:
            continue
        verb_str = ' <sep> '.join(verbs)
        u_name, v_name = edge_data['u_name'], edge_data['v_name']
        relationship = '<s> {} <v> {} <o> {}'.format(
            u_name,
            verb_str,
            v_name
        )
        output.append(relationship)
    return ' '.join(output), max_overlap, target_node_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PyTorch Dataset wrapper for Question Generation Task.')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    dtypes = ['mini']
    dataset = dataset_factory(args.dataset)
    for dtype in dtypes:
        print('Linearizing knowledge graphs for {} set'.format(dtype))
        d = dataset[dtype]
        kg_fn = os.path.join('..', 'data', dataset.name, 'kg_{}.pk'.format(dtype))
        with open(kg_fn, 'rb') as fd:
            kgs = pickle.load(fd)

        output = []
        for i in range(len(d)):
            example = d[i]
            id = example['_id']
            graph = kgs[id]
            q_toks = tokenize(example['question'])
            a_toks = tokenize(example['answer'])
            graph_seq, max_overlap, target_node_idx = linearize_graph(graph, a_toks)

            output.append({
                'qid': id,
                'graph_seq': graph_seq,
                'q_toks': q_toks,
                'a_toks': a_toks,
                'overlap': max_overlap,  # answer-node overlap
                'target_node_idx': target_node_idx,
            })

        out_fn = os.path.join('..', 'data', dataset.name, 'dataset_{}.csv'.format(dtype))
        df = pd.DataFrame(output)
        df.to_csv(out_fn, index=False)
