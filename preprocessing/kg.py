from collections import Counter, defaultdict
from functools import partial
from itertools import chain
import json
from multiprocessing import Manager, Pool
import pickle
import os
import sys
from string import punctuation
from time import time

import argparse
import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import spacy

from dataset_base import dataset_factory
from utils import duration

DIST_THRESHOLD = 0.2


def tok_tup(tups):
    return list(map(lambda tup: list(map(lambda x: ' '.join(tokenize(x)), tup)), tups))


def tokenize(str):
    toks = [token.text.lower().strip(punctuation).strip() for token in spacy_nlp(str)]
    return [tok for tok in toks if len(tok) > 0]


def _construct_graph(oie_tuples, node_assignments, cluster_assignments, node_weights, head_names):
    g = nx.MultiDiGraph()
    n = len(cluster_assignments)
    for i in range(n):
        g.add_node(i, head=head_names[i], aliases=cluster_assignments[i], weight=node_weights[i])

    for tup in oie_tuples:
        u, e, v = tup
        u_node = node_assignments[u]
        v_node = node_assignments[v]
        g.add_edge(u_node, v_node, name=e)
    return g


def construct_graph(input, lock=None, ctr=None):
    qid, context_ids = input
    oie_tuples = [oie_data_tok[cid] for cid in context_ids]
    oie_tuples_flat = list(chain(*oie_tuples))
    nodes = []
    for tup in oie_tuples_flat:
        nodes += [tup[0], tup[2]]
    node_counts = Counter(nodes)
    nodes_uniq = list(node_counts.keys())

    if len(nodes_uniq) == 0:
        print('{} id has no IE tuples'.format(qid))
        return 'N/A'

    inputs = tf_idf_vectorizer.transform(nodes_uniq)
    s = inputs.sum(1)
    zero_tf_idxs = np.where(s == 0)[0]
    nonzero_tf_idxs = np.where(s > 0)[0]

    valid_nodes = [nodes_uniq[idx] for idx in nonzero_tf_idxs]
    valid_inputs = inputs[nonzero_tf_idxs]

    invalid_nodes = [nodes_uniq[idx] for idx in zero_tf_idxs]
    clusterizer = AgglomerativeClustering(n_clusters=None, affinity='cosine', distance_threshold=DIST_THRESHOLD,
                                          linkage='average', compute_full_tree=True)

    cluster_labels = clusterizer.fit_predict(valid_inputs.toarray())
    node_assignments = defaultdict(int)
    max_label = 0
    for node, label in zip(valid_nodes, cluster_labels):
        label = int(label)
        node_assignments[node] = label
        max_label = max(max_label, label)
    offset = max_label + 1
    for idx in range(len(invalid_nodes)):
        node_assignments[invalid_nodes[idx]] = idx + offset
    cluster_assignments = defaultdict(list)
    for node, cluster in node_assignments.items():
        cluster_assignments[cluster].append(node)

    cluster_counts = {}
    head_names = {}
    for cluster, names in cluster_assignments.items():
        counts = [node_counts[n] for n in names]
        max_count = max(counts)
        cluster_counts[cluster] = sum(counts)
        candidates = [names[i] for i, count in enumerate(counts) if count == max_count]
        candidate_lens = [len(x) for x in candidates]
        head_name = candidates[np.argmax(candidate_lens)]
        head_names[cluster] = head_name

    g = _construct_graph(oie_tuples_flat, node_assignments, cluster_assignments, cluster_counts, head_names)
    out_fn = os.path.join(data_dir, 'chunks', '{}.pk'.format(qid))
    print('Dumping knowledge graphs to {}'.format(out_fn))
    with open(out_fn, 'wb') as fd:
        pickle.dump(g, fd)
    with lock:
        ctr.value += 1
        if ctr.value % update_incr == 0:
            print('Processed {} contexts...'.format(ctr.value))
    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate knowledge graphs from OIE6 output.')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    dataset = dataset_factory(args.dataset)
    print('Loading Spacy...')
    spacy_nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner', 'tagger', 'textcat'])

    data_dir = os.path.join('..', 'data', dataset.name)
    tf_idf_fn = os.path.join(data_dir, 'tf_idf_vectorizer.pk')
    print('Loading TF-IDF vectorizer...')
    with open(tf_idf_fn, 'rb') as fd:
        tf_idf_vectorizer = pickle.load(fd)

    update_incr = 10 if args.debug else 100
    dtypes = ['mini'] if args.debug else ['train']  # , 'test', 'validation']
    results = []
    for dtype in dtypes:
        start_time = time()
        oie_fn = os.path.join(data_dir, 'open_ie_data', 'predictions_{}.json'.format(dtype))
        print('Loading open IE output for {} set...'.format(dtype))
        with open(oie_fn, 'r') as fd:
            oie_data = json.load(fd)

        print('Tokenizing tuple output...')
        oie_data_items = oie_data.items()
        keys = [a[0] for a in oie_data_items]
        all_tups = [a[1] for a in oie_data_items]
        with Manager() as manager:
            p = Pool()
            tok_vals = list(p.map(tok_tup, all_tups))
            p.close()
            p.join()
        oie_data_tok = dict(zip(keys, tok_vals))

        print('Loading contexts...')
        id_context_map = dataset.get_linked_contexts(dtype).items()
        ids = [id[0] for id in id_context_map]
        print('Starting to construct graphs...')
        with Manager() as manager:
            p = Pool()
            lock = manager.Lock()
            ctr = manager.Value('i', 0)
            graphs = list(p.map(partial(construct_graph, lock=lock, ctr=ctr), id_context_map))
            p.close()
            p.join()
        duration(start_time)

        graphs = dict(zip(ids, graphs))
        out_fn = os.path.join(data_dir, 'kg_{}.pk'.format(dtype))
        print('Dumping knowledge graphs to {}'.format(out_fn))
        with open(out_fn, 'wb') as fd:
            pickle.dump(graphs, fd)
        print('Done!')
