import json
import os
import pickle
from string import punctuation

import argparse
import networkx as nx
from networkx.algorithms.traversal.breadth_first_search import bfs_edges
import pandas as pd
import spacy
from p_tqdm import p_uimap

from dataset_base import dataset_factory
from kg import tokenize

TUP = '<tup>'
ARG1 = '<a1>'
REL = '<r>'
ARG2 = '<a2>'
SEP = '<sep>'
HL = '<hl>'
SPECIAL_TOKS = [
    ARG1,
    REL,
    ARG2,
    HL
]

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


def highlight(name, highlight_toks, threshold=None, mask=True):
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
    if max_overlap >= threshold:
        ans_arr = [HL] if mask else ([HL] + name_toks[s:e] + [HL])
        return ' '.join(name_toks[:s] + ans_arr + name_toks[e:]), True
    return name, False


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
        return '', 0.0, -1, 0, '', '', '', '', ''

    output = []
    edge_list = []
    edge_list_resolved = []
    shortest_paths = nx.shortest_path_length(graph, source=None, target=target_node_idx)
    assert shortest_paths[target_node_idx] == 0
    weights = []
    node_list = []
    node_dist = []
    num_edges = 0
    assigned_hl = False
    for i, edge in enumerate(bfs_edges(graph, target_node_idx)):
        edge_data = graph.edges[edge]
        verbs = edge_data['verbs']
        verbs -= {'subset'}
        verbs = list(verbs)
        if len(verbs) == 0:
            continue
        verb_str = (' {} '.format(SEP)).join(verbs)
        u_name, v_name = edge_data['u_name'], edge_data['v_name']
        assigned = False

        if edge_data['back_edge']:
            v, u = edge
            edge_r = [v, u]
        else:
            u, v = edge
            edge_r = [u, v]

        u_meta = graph.nodes[u]
        v_meta = graph.nodes[v]

        node_list += [u, v]
        node_dist += [shortest_paths[u], shortest_paths[v]]

        u_weight = u_meta['weight']
        v_weight = v_meta['weight']
        edge_weight = (u_weight + v_weight) / 2.0
        if edge_data['back_edge'] and not assigned_hl:
            v_name, assigned = highlight(v_name, answer_tok_set, threshold=max_overlap)
            assigned_hl = assigned_hl or assigned
        if not edge_data['back_edge'] and not assigned_hl:
            u_name, assigned = highlight(u_name, answer_tok_set, threshold=max_overlap)
            assigned_hl = assigned_hl or assigned

        relationship = '{} {} {} {} {} {} {}'.format(
            TUP,
            ARG1,
            u_name,
            REL,
            verb_str,
            ARG2,
            v_name
        )

        weight_arr = [edge_weight] * len(relationship.split(' '))
        if assigned:
            output.insert(0, relationship)
            edge_list.insert(0, edge)
            edge_list_resolved.insert(0, edge_r)
            weights = weight_arr + weights
        else:
            output.append(relationship)
            edge_list.append(edge)
            edge_list_resolved.append(edge_r)
            weights += weight_arr
        num_edges += 1
    output_str = ' '.join(output)
    return (
        output_str, max_overlap, target_node_idx, num_edges, edge_list, edge_list_resolved, weights, node_list,
        node_dist
    )


def linearize_graph(input):
    example, graph = input

    id = example[qid_key]
    q_toks = tokenize(example[q_key])
    ans = None
    for k in a_keys:
        if ans is None:
            ans = example[k]
        else:
            ans = ans[k]

    a_toks = tokenize(ans)
    (graph_seq, max_overlap, target_node_idx, num_edges, edge_list, edge_list_resolved, weights, node_list,
     node_dists
     ) = _linearize_graph(graph, a_toks)

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
        'num_edges': num_edges,
        'edge_list': json.dumps(edge_list),
        'edge_list_resolved': json.dumps(edge_list_resolved),
        'weights': json.dumps(weights),
        'nodes': json.dumps(node_list),
        'node_dists': json.dumps(node_dists)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PyTorch Dataset wrapper for Question Generation Task.')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    parser.add_argument('--dtypes', default=None)
    args = parser.parse_args()

    dataset = dataset_factory(args.dataset)

    qid_key = dataset.qid_key()
    q_key = dataset.question_key()
    a_keys = dataset.answer_keys()

    if args.dtypes is None:
        dtypes = ['mini'] if args.debug else ['validation', 'train']
        if not dataset.name == 'squad' and not args.debug:
            dtypes.append('test')
    else:
        dtypes = args.dtypes.split(',')

    for dtype in dtypes:
        print('Linearizing knowledge graphs for {} set'.format(dtype))
        d = dataset[dtype]
        kg_fn = os.path.join('..', 'data', dataset.name, 'kg_{}.pk'.format(dtype))
        print('Loading knowledge graphs for {} set...'.format(dtype))
        with open(kg_fn, 'rb') as fd:
            kgs = pickle.load(fd)

        n = len(d)
        examples_w_graph = [(example, kgs[example[qid_key]]) for example in d if example[qid_key] in kgs]
        outputs = list(p_uimap(linearize_graph, examples_w_graph))

        out_fn = os.path.join('..', 'data', dataset.name, 'dataset_{}.csv'.format(dtype))
        df = pd.DataFrame(outputs)
        print('Saving {} examples to {}'.format(df.shape[0], out_fn))
        df.to_csv(out_fn, index=False)

        viable_out_fn = os.path.join('..', 'data', dataset.name, 'dataset_viable_{}.csv'.format(dtype))
        df.dropna(inplace=True)
        min_viable_q_recall = 0.5
        min_viable_edges = 2
        df = df[(df['q_tok_recall'] >= min_viable_q_recall) & (df['num_edges'] >= min_viable_edges)]
        print('Saving {} examples to {}'.format(df.shape[0], out_fn))
        df.to_csv(viable_out_fn, index=False)
