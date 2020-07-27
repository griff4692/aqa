from collections import Counter
import os
import pickle
from string import punctuation
import sys
import unicodedata

import argparse
import pandas as pd
import spacy
from torch.utils.data import Dataset
from tqdm import tqdm
from p_tqdm import p_uimap


from dataset_base import dataset_factory
from kg import tokenize

import networkx as nx
from networkx.algorithms.traversal.breadth_first_search import bfs_edges

print('Loading Spacy...')
spacy_nlp = spacy.load('en_core_web_lg')
print('Finished loading Spacy...')
STOPWORDS = set(spacy.lang.en.stop_words.STOP_WORDS)


def tokenize(str):
    toks = [token.text.lower().strip(punctuation).strip() for token in spacy_nlp(str)]
    return [tok for tok in toks if len(tok) > 0]


def overlap(a, b):
    avg_len = float(len(a) + len(b)) / 2.0
    c = float(len(a.intersection(b)))
    return c / avg_len


def highlight(name, highlight_toks):
    name_toks = tokenize(name)
    n = len(name_toks)
    s, e = None, None
    max_overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n + 1):
            o = overlap(set(name_toks[i:j]), highlight_toks)
            if o >= max_overlap:
                max_overlap = o
                s = i
                e = j
    return ' '.join(name_toks[:s] + ['<hl>'] + name_toks[s:e] + ['<hl>'] + name_toks[e:])


def _linearize_graph(graph, answer_toks):
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
        return '', 0.0, -1, 0

    output = []
    num_edges = 0
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
        if i == 0 and edge_data['back_edge']:
            v_name = highlight(v_name, answer_tok_set)
        if i == 0 and not edge_data['back_edge']:
            u_name = highlight(u_name, answer_tok_set)

        relationship = '<s> {} <v> {} <o> {}'.format(
            u_name,
            verb_str,
            v_name
        )
        output.append(relationship)
        num_edges += 1
    return ' '.join(output), max_overlap, target_node_idx, num_edges


def linearize_graph(input):
    example, graph = input

    id = example['id']
    q_toks = tokenize(example['question'])
    a_toks = tokenize(example['answers']['text'][0])
    graph_seq, max_overlap, target_node_idx, num_edges = _linearize_graph(graph, a_toks)

    graph_tok_set = set(graph_seq.split(' '))
    q_tok_set = set(q_toks) - STOPWORDS
    q_tok_recall = len(q_tok_set.intersection(graph_tok_set)) / max(1.0, float(len(q_tok_set)))

    return {
        'qid': id,
        'graph_seq': graph_seq,
        'q_toks': ' '.join(q_toks),
        'a_toks': ' '.join(a_toks),
        'overlap': max_overlap,  # answer-node overlap
        'target_node_idx': target_node_idx,
        'q_tok_recall': q_tok_recall,
        'num_edges': num_edges
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PyTorch Dataset wrapper for Question Generation Task.')
    parser.add_argument('--dataset', default='squad', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    dataset = dataset_factory(args.dataset)
    if dataset.name == 'squad':
        dtypes = ['mini'] if args.debug else ['train', 'validation']
    else:
        dtypes = ['mini'] if args.debug else ['train', 'test', 'validation']

    for dtype in dtypes:
        print('Linearizing knowledge graphs for {} set'.format(dtype))
        d = dataset[dtype]
        kg_fn = os.path.join('..', 'data', dataset.name, 'kg_{}.pk'.format(dtype))
        print('Loading knowledge graphs for {} set...'.format(dtype))
        with open(kg_fn, 'rb') as fd:
            kgs = pickle.load(fd)

        n = len(d)
        examples_w_graph = [(example, kgs[example['id']]) for example in d if example['id'] in kgs]
        outputs = list(p_uimap(linearize_graph, examples_w_graph))

        out_fn = os.path.join('..', 'data', dataset.name, 'dataset_{}.csv'.format(dtype))
        df = pd.DataFrame(outputs)
        print('Saving {} examples to {}'.format(df.shape[0], out_fn))
        df.to_csv(out_fn, index=False)
