from collections import Counter, defaultdict
import itertools
import json
from multiprocessing import Manager, Pool
import pickle
import os
import shutil
from string import punctuation
from time import time
from tqdm import tqdm
import toyplot
import toyplot.pdf as to_pdf

import argparse
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from dataset_base import dataset_factory
from utils import duration


def format_text(str):
    tokens = str.split(' ')
    token_br = []
    curr_len = 0
    for token in tokens:
        token_br.append(token)
        curr_len += len(token)
        if curr_len > 25:
            token_br.append('<br/>')
            curr_len = 0
    return ' '.join(token_br)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate knowledge graphs from OIE6 output.')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    dataset = dataset_factory(args.dataset)

    with open('../data/hotpot_qa/kg_mini.pk', 'rb') as fd:
        x = pickle.load(fd)

    G = x[list(x.keys())[0]]

    # node_data = defaultdict(list)
    # edge_data = defaultdict(list)

    # remove nodes which only have a 'subset' edge
    # for node, meta in G.nodes(data=True):
    #     node_data['head'].append(meta['head'])
    #     node_data['Id'].append(node)
    #     node_data['aliases'].append('|'.join(meta['aliases']))
    #     node_data['weight'].append(meta['weight'])

    # for u, v, meta in G.edges(data=True):
    #     if meta['back_edge']:
    #         continue
    #     edge_data['Source'].append(u)
    #     edge_data['Target'].append(v)
    #     edge_data['Type'].append('directed')
    #     edge_data['verbs'].append('|'.join(list(meta['verbs'])))
    # 
    # node_df = pd.DataFrame(node_data)
    # edge_df = pd.DataFrame(edge_data)
    # 
    # node_df.to_csv('node.csv', index=False)
    # edge_df.to_csv('edge.csv', index=False)
    
    edges = []
    
    for u, v, meta in G.edges(data=True):
        if meta['back_edge']:
            continue

        verbs = list(meta['verbs'])
        is_sub = len(verbs) == 1 and verbs[0] == 'subset'
        dead_end = G.degree[u] <= 2 and is_sub
        if dead_end:
            continue

        u_head = G.nodes[u]['head']
        v_head = G.nodes[v]['head']

        edge = [
            format_text(u_head),
            format_text(v_head)
        ]
        edges.append(edge)

    canvas, axes, mark = toyplot.graph(np.array(edges), vsize=30, width=5000, height=5000)
    to_pdf.render(canvas, 'figure1.pdf')
