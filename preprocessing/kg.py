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

import argparse
import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from torch.nn import DataParallel as DP

from dataset_base import dataset_factory
from utils import duration

DIST_THRESHOLD = 0.25


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def tok_tup(tups):
    return list(map(lambda tup: list(map(lambda x: ' '.join(tokenize(x)), tup)), tups))


def tokenize(str):
    toks = [token.text.lower().strip(punctuation).strip() for token in spacy_tokenizer(str)]
    return [tok for tok in toks if len(tok) > 0]


def _construct_graph(oie_tuples, node_assignments, cluster_assignments, node_weights, head_names):
    g = nx.DiGraph()
    n = len(cluster_assignments)
    for i in range(n):
        g.add_node(i, head=head_names[i], aliases=cluster_assignments[i], weight=node_weights[i])

    for tup in oie_tuples:
        u, e, v = tup
        u_node = node_assignments[u]
        v_node = node_assignments[v]

        if u_node == v_node:
            continue  # no self loops

        if g.has_edge(u_node, v_node):
            g.edges[u_node, v_node]['verbs'].add(e)
            g.edges[v_node, u_node]['verbs'].add(e)
        else:
            e_set = {e}
            g.add_edge(u_node, v_node, u_name=u, v_name=v, verbs=e_set, back_edge=False)
            g.add_edge(v_node, u_node, u_name=u, v_name=v, verbs=e_set, back_edge=True)
    return g


def generate_pairs(n):
    return itertools.combinations(range(n), r=2)


def construct_graph(input, out_dir=None):
    qid, context_ids = input
    oie_tuples = [oie_data_tok[cid] for cid in context_ids]
    oie_tuples_flat = list(itertools.chain(*oie_tuples))
    oie_tuples_flat = [[x.strip() for x in tup] for tup in oie_tuples_flat]
    nodes = []
    for tup in oie_tuples_flat:
        nodes += [tup[0], tup[2]]
    node_counts = Counter(nodes)
    nodes_uniq = set(list(node_counts.keys()))

    # add noun chunks with special 'subset' predicate relationship
    for node in nodes_uniq.copy():
        if len(node.split(' ')) > 1:
            for chunk in chunker(node).noun_chunks:
                chunk = chunk.string.strip()
                if len(chunk) > 1:
                    oie_tuples_flat.append([chunk, 'subset', node])
                    nodes_uniq.add(chunk)

    if len(nodes_uniq) == 0:
        print('{} id has no IE tuples'.format(qid))
        return 'N/A'

    nodes_uniq = list(nodes_uniq)
    n = len(nodes_uniq)
    mat_idxs = list(generate_pairs(n))

    rs, cs = [], []
    for r, c in mat_idxs:
        rs.append(nodes_uniq[r])
        cs.append(nodes_uniq[c])

    sim_numbers = compute_sim(rs, cs)
    dist_mat = np.ones([n, n], dtype=float)
    assert len(sim_numbers) == len(mat_idxs)
    for sim, (r, c) in zip(sim_numbers, mat_idxs):
        dist_mat[r, c] = dist_mat[c, r] = 1.0 - sim

    clusterizer = AgglomerativeClustering(n_clusters=None, affinity='precomputed', distance_threshold=DIST_THRESHOLD,
                                          linkage='average', compute_full_tree=True)

    cluster_labels = clusterizer.fit_predict(dist_mat)
    node_assignments = defaultdict(int)
    for node, label in zip(nodes_uniq, cluster_labels):
        label = int(label)
        node_assignments[node] = label

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
    out_fn = os.path.join(out_dir, '{}.pk'.format(qid))
    output = {qid: g}
    with open(out_fn, 'wb') as fd:
        pickle.dump(output, fd)
    return qid, g


def compute_sim(x, y):
    input = list(zip(x, y))
    results = []

    if torch.cuda.is_available():
        batch_size = 200 * len(model.device_ids)
        device_str = f'cuda:{model.device_ids[0]}'
    else:
        batch_size = 200
        device_str = 'cpu'

    for chunk in chunks(input, batch_size):
        x = [c[0] for c in chunk]
        y = [c[1] for c in chunk]
        batch_input = tokenizer(
            x, y, return_tensors='pt', max_length=20, padding='max_length', truncation=True).to(device_str)
        logits = model(**batch_input)[0]
        batch_results = torch.softmax(logits, dim=1)[:, 1].tolist()
        results += batch_results
    return results


def load_precomputed(out_dir, output):
    for fn in os.listdir(out_dir):
        chunk_in_fn = os.path.join(out_dir, fn)
        with open(chunk_in_fn, 'rb') as fd:
            o = pickle.load(fd)
            output.update(o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate knowledge graphs from OIE6 output.')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    dataset = dataset_factory(args.dataset)
    print('Loading Spacy...')
    spacy_tokenizer = spacy.load('en_core_web_lg', disable=['parser', 'ner', 'tagger', 'textcat'])
    chunker = spacy.load('en_core_web_lg')

    data_dir = os.path.join('..', 'data', dataset.name)
    tmp = []
    print('Loading ALBERT...')

    #  getting the list of GPUs available
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        gpus = len(device_ids)
        print('GPU detected')
    else:
        DEVICE = torch.device("cpu")
        device_ids = -1
        print('No GPU. switching to CPU')

    model = 'textattack/albert-base-v2-MRPC'
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)
    if not device_ids == -1:
        print('Porting model to CUDA...')
        model = DP(model, device_ids=device_ids)
        model.to(f'cuda:{model.device_ids[0]}')
    model.eval()

    update_incr = 10 if args.debug else 100
    if dataset.name == 'squad':
        dtypes = ['mini'] if args.debug else ['validation', 'train']
    else:
        dtypes = ['mini'] if args.debug else ['test', 'validation', 'train']

    results = []
    for dtype in dtypes:
        start_time = time()
        oie_fn = os.path.join(data_dir, 'oie_data', 'predictions_{}.json'.format(dtype))
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

        print('Starting to construct graphs...')
        out_dir = os.path.join(data_dir, 'chunks', 'kg', dtype)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        output = {}
        load_precomputed(out_dir, output)
        already_n = len(output)

        id_context_map_todo = [item for item in id_context_map if not item[0] in output]
        n = len(id_context_map_todo)
        print('Already computed {} outputs.  Doing {} more...'.format(already_n, n))
        for i in tqdm(range(n)):
            construct_graph(id_context_map_todo[i], out_dir=out_dir)
        load_precomputed(out_dir, output)
        duration(start_time)
        out_fn = os.path.join(data_dir, 'kg_{}.pk'.format(dtype))
        print('Dumping {} knowledge graphs to {}'.format(len(output), out_fn))
        with open(out_fn, 'wb') as fd:
            pickle.dump(output, fd)
        print('Done!  Now can safely remove cached')
        print('Removing temporary chunks...')
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
