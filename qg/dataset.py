import os
import pickle
from string import punctuation
import sys
import unicodedata
sys.path.insert(0, os.path.expanduser('~/aqa/preprocessing'))

import argparse
import spacy
from torch.utils.data import Dataset

from dataset_base import dataset_factory
from kg import tokenize

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
    max_overlap = 0.0
    for node in node_objects:
        node_idx, metadata = node
        head_tok_set = set(metadata['head'].split(' '))
        o = overlap(answer_tok_set, head_tok_set)
        if o > max_overlap:
            target_node_idx = node_idx
            max_overlap = o
    path = [x for x in bfs_edges(graph, target_node_idx)]
    print(graph.nodes[target_node_idx]['head'], answer_toks)
    return path


class QGD(Dataset):
    def __init__(self, dataset_str, dtype):
        self.dataset = dataset_factory(dataset_str)[dtype]
        kg_fn = os.path.join('..', 'data', dataset_str, 'kg_{}.pk'.format(dtype))
        with open(kg_fn, 'rb') as fd:
            self.kgs = pickle.load(fd)

    def __getitem__(self, item):
        example = self.dataset[item]
        id = example['_id']
        graph = self.kgs[id]
        q_toks = tokenize(example['question'])
        a_toks = tokenize(example['answer'])
        graph_seq = linearize_graph(graph, a_toks)
        return graph_seq, q_toks, a_toks

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PyTorch Dataset wrapper for Question Generation Task.')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()
    dataset = dataset_factory(args.dataset)
    dataset = QGD(args.dataset, 'mini')
    for i in range(len(dataset)):
        x = dataset[i]
