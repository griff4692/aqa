from collections import defaultdict
import re

import numpy as np

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
    if is_tgt:
        assert len(dependencies) == 1
        return resolved[list(dependencies)[0]]
    span = str2span(span_str)
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
            toks += resolved[copy_str]
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

    for cluster in clusters:
        cluster_toks = list(map(lambda x: document[x[0]:x[1] + 1], cluster))
        spans_no_pronouns = map(lambda x: set(x) - PRONOUNS, cluster_toks)
        span_lens = np.array(list(map(len, spans_no_pronouns)))
        head_span_idx = None
        for i, span_len in enumerate(span_lens):
            if span_len > 0:
                head_span_idx = i
                break

        head_span = cluster[head_span_idx]
        head_span_str = span2str(head_span)
        head_name = ' '.join(cluster_toks[head_span_idx])
        for i, span in enumerate(cluster):
            if i == head_span_idx:
                continue
            tgt_span_str = span2str(span)
            tgt_span_toks = cluster_toks[i]
            is_contained = re.search(r'{}'.format(head_name), ' '.join(tgt_span_toks)) is not None

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
    example = {
        'document': [
            'Paul', 'Allen', 'was', 'born', 'on', 'January', '21', ',', '1953', ',', 'in', 'Seattle', ',', 'Washington',
            ',', 'to', 'Kenneth', 'Sam', 'Allen', 'and', 'Edna', 'Faye', 'Allen', '.', 'Allen', 'attended', 'Lakeside',
            'School', ',', 'a', 'private', 'school', 'in', 'Seattle', ',', 'where', 'he', 'befriended', 'Bill',
            'Gates', ',', 'two', 'years', 'younger', ',', 'with', 'whom', 'he', 'shared', 'an', 'enthusiasm', 'for',
            'computers', '.', 'Paul', 'and', 'Bill', 'used', 'a', 'teletype', 'terminal', 'at', 'their', 'high',
            'school', ',', 'Lakeside', ',', 'to', 'develop', 'their', 'programming', 'skills', 'on', 'several',
            'time', '-', 'sharing', 'computer', 'systems',  '.'
        ],
        'clusters': [
            [[0, 1], [24, 24], [36, 36], [47, 47], [54, 54]],
            [[11, 14], [33, 33]],
            [[38, 52], [56, 56]],
            [[54, 56], [62, 62], [70, 70]],
            [[26, 34], [62, 67]]
        ]
    }

    resolved_toks = resolve(example['document'], example['clusters'])
    print(' '.join(resolved_toks))
