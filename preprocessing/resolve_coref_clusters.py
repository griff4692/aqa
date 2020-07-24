from collections import defaultdict
import json
import os
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import numpy as np
from time import time

from dataset_base import dataset_factory
from utils import duration

PRONOUNS = {
    'all', 'another', 'any', 'anybody', 'anyone', 'anything', 'as', 'aught', 'both', 'each other', 'each', 'either',
    'enough', 'everybody', 'everyone', 'everything', 'few', 'he', 'her', 'hers', 'herself', 'him', 'himself', 'his',
    'i', 'idem', 'it', 'its', 'itself', 'many', 'me', 'mine', 'most', 'my', 'myself', 'naught', 'neither', 'no one',
    'nobody', 'none', 'nothing', 'nought', 'one another', 'one', 'other', 'others', 'ought', 'our', 'ours', 'ourself',
    'ourselves', 'several', 'she', 'some', 'somebody', 'someone', 'something', 'somewhat', 'such', 'suchlike', 'that',
    'thee', 'their', 'theirs', 'theirself', 'theirselves', 'them', 'themself', 'themselves', 'there', 'these', 'they',
    'thine', 'this', 'those', 'thou', 'thy', 'thyself', 'us', 'we', 'what', 'whatever', 'whatnot', 'whatsoever',
    'whence', 'where', 'whereby', 'wherefrom', 'wherein', 'whereinto', 'whereof', 'whereon', 'wheresoever', 'whereto',
    'whereunto', 'wherever', 'wherewith', 'wherewithal', 'whether', 'which', 'whichever', 'whichsoever', 'who',
    'whoever', 'whom', 'whomever', 'whomso', 'whomsoever', 'whose', 'whosesoever', 'whosever', 'whoso', 'whosoever',
    'ye', 'yon', 'yonder', 'you', 'your', 'yours', 'yourself', 'yourselves'
}


def span2str(span):
    return '{}_{}'.format(str(span[0]), str(span[1]))


def span2toks(span, d):
    return d[span[0]:span[1] + 1]


def str2span(str):
    s, e = str.split('_')
    return [int(s), int(e)]


def is_subset(a, b):
    return a[0] >= b[0] and a[1] <= b[1]


def span2len(span):
    return span[1] - span[0] + 1


def build_doc(resolved, spans, d):
    curr_idx = 0
    toks = []
    while curr_idx < len(d):
        copy_to = -1
        for span in spans:
            if span[0] == curr_idx:
                copy_to = span[1]
                break

        if copy_to > -1:
            copy_str = span2str([curr_idx, copy_to])
            toks += resolved[copy_str]
            curr_idx = copy_to + 1
        else:
            toks.append(d[curr_idx])
            curr_idx += 1
    return toks


def build_str(span_str, resolved, dependencies, d, is_tgt):
    span = str2span(span_str)
    if is_tgt:
        copy_span_str = list(dependencies)[0]
        if copy_span_str in resolved:
            return resolved[copy_span_str]
        else:
            print('Warning. Circular dependency detected!')
            copy_span = str2span(copy_span_str)
            return d[copy_span[0]:copy_span[1] + 1]
    s = span[0]
    e = span[1]
    # remove sub-dependencies
    dep_spans = list(map(str2span, dependencies))
    dep_span_lens = list(map(span2len, dep_spans))
    dep_order = np.argsort(dep_span_lens)

    toks = []
    curr_idx = s
    while curr_idx <= e:
        copy_to = -1
        for dep_idx in dep_order:
            if dep_spans[dep_idx][0] == curr_idx:
                copy_to = dep_spans[dep_idx][1]
        if copy_to > -1:
            copy_str = span2str([curr_idx, copy_to])
            if copy_str in resolved:
                toks += resolved[copy_str]
            else:
                print('Warning. Circular dependency likely. Skipping.')
            curr_idx = copy_to + 1
        else:
            toks.append(d[curr_idx])
            curr_idx += 1

    return toks


def resolve(document, clusters):
    all_spans = set()

    span_dependencies = defaultdict(set)
    span_dependencies_rev = defaultdict(set)
    dep_counts = defaultdict(int)
    replaced_span_strs = set()
    subsumed_set = set()

    clusters_non_overlapping = []
    for cluster in clusters:
        cluster_starts = [c[0] for c in cluster]
        cluster_order = np.argsort(np.array(cluster_starts))

        cluster_non_overlapping = []
        for i, cidx in enumerate(cluster_order):
            curr_cluster = cluster[cidx]
            for j in range(i + 1, len(cluster)):
                if curr_cluster[1] >= cluster[cluster_order[j]][0]:
                    e_idx = max(curr_cluster[0], cluster[cluster_order[j]][0] - 1)
                    curr_cluster = [curr_cluster[0], e_idx]
                    break
            cluster_non_overlapping.append(curr_cluster)
        clusters_non_overlapping.append(cluster_non_overlapping)

        cluster_toks = list(map(lambda x: document[x[0]:x[1] + 1], cluster_non_overlapping))
        spans_no_pronouns = map(lambda x: set([y.lower() for y in x]) - PRONOUNS, cluster_toks)
        span_lens = np.array(list(map(len, spans_no_pronouns)))
        head_span_idx = None
        for i, span_len in enumerate(span_lens):
            if span_len > 0:
                head_span_idx = i
                break

        if head_span_idx is None:
            head_span_idx = 0

        head_span = cluster_non_overlapping[head_span_idx]
        head_span_str = span2str(head_span)
        head_name = ' '.join(cluster_toks[head_span_idx])
        for i, span in enumerate(cluster_non_overlapping):
            if i == head_span_idx:
                continue
            tgt_span_str = span2str(span)
            if tgt_span_str in replaced_span_strs:
                continue

            tgt_span_toks = cluster_toks[i]
            is_contained = re.search(re.escape(head_name.lower()), ' '.join(tgt_span_toks).lower()) is not None
            if is_contained:
                continue
            all_spans.add(head_span_str)
            all_spans.add(tgt_span_str)

            span_dependencies[tgt_span_str].add(head_span_str)
            span_dependencies_rev[head_span_str].add(tgt_span_str)
            dep_counts[tgt_span_str] = 1
            replaced_span_strs.add(tgt_span_str)

    all_spans = list(all_spans)
    all_span_lens = list(map(lambda x: span2len(str2span(x)), all_spans))
    order = np.argsort(np.array(all_span_lens))

    for i, span_idx in enumerate(order):
        small_span_str = all_spans[span_idx]
        small_span = str2span(small_span_str)
        for j in range(i, len(order)):
            other_span_idx = order[j]
            if other_span_idx == span_idx:
                continue
            big_span_str = all_spans[other_span_idx]
            big_span = str2span(big_span_str)
            if big_span_str in replaced_span_strs:
                continue
            is_sub = is_subset(small_span, big_span)
            if is_sub:
                subsumed_set.add(small_span_str)
            if is_sub and not big_span_str in replaced_span_strs:
                span_dependencies[big_span_str].add(small_span_str)
                span_dependencies_rev[small_span_str].add(big_span_str)
                dep_counts[big_span_str] += 1

    resolved = {}
    for _ in range(len(all_spans)):
        min_dep = 100000
        min_span = None
        for span in all_spans:
            dc = dep_counts[span]
            if dc <= min_dep and span not in resolved:
                min_dep = dc
                min_span = span
        is_tgt = min_span in replaced_span_strs
        resolved[min_span] = build_str(min_span, resolved, span_dependencies[min_span], document, is_tgt)
        for child_span in span_dependencies_rev[min_span]:
            dep_counts[child_span] -= 1

    to_replace_span_strs = list(map(str2span, set(all_spans) - subsumed_set))
    resolved_toks = build_doc(resolved, to_replace_span_strs, document)
    return resolved_toks


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate json file for  consumption by SpanBERT.')
    parser.add_argument('--dataset', default='squad', help='squad, trivia_qa, or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    data_dir = os.path.join('..', 'data', args.dataset, 'coref_data')
    dataset = dataset_factory(args.dataset)
    if dataset.name == 'squad':
        dtypes = ['mini'] if args.debug else ['train', 'validation']
    else:
        dtypes = ['mini'] if args.debug else ['train', 'test', 'validation']

    for dtype in dtypes:
        start_time = time()
        keys_fn = os.path.join(data_dir, 'coref_keys_{}.json'.format(dtype))
        coref_fn = os.path.join(data_dir, 'coref_output_{}.json'.format(dtype))
        with open(coref_fn, 'r') as fd:
            corefs = json.load(fd)
        with open(keys_fn, 'r') as fd:
            keys = json.load(fd)

        from tqdm import tqdm
        resolved = list(tqdm(map(
            lambda coref_output: resolve(coref_output['document'], coref_output['clusters']),
            corefs
        ), total=len(corefs)))

        output = {}
        for i in range(len(keys)):
            k, v = keys[i], corefs[i]
            v['resolved'] = resolved[i]
            output[k] = v

        duration(start_time)
        out_fn = os.path.join(data_dir, 'coref_resolved_{}.json'.format(dtype))
        print('Dumping to {}'.format(out_fn))

        with open(out_fn, 'w') as fd:
            json.dump(output, fd)
