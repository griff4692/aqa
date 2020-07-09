from collections import defaultdict
from functools import partial
import json
from multiprocessing import Manager, Pool
import os
from time import time

import argparse
import neuralcoref
import spacy
import torch

from dataset_base import dataset_factory
from utils import duration, remove_extra_space


def construct_coref(t):
    """
    :param t: string or list of strings
    :return: a dictionary consisting of 'clusters', 'resolved' where 'resolved' is the output of replacing coreferent
    entity 'clusters' with their head (canonical) term.
    """
    coref = coref_nlp(t)._
    clusters = defaultdict(set)
    cluster_set = coref.coref_clusters
    for cluster in cluster_set:
        mentions = list(map(lambda x: remove_extra_space(x.text.lower().strip()), cluster.mentions))
        clusters[remove_extra_space(cluster.main.text.lower().strip())].update(mentions)

    # Can't JSON serialize a set
    for k, v in clusters.items():
        clusters[k] = list(v)

    obj = {
        'clusters': clusters,
        'resolved': coref.coref_resolved.strip()
    }

    return obj


def process_context(t, lock=None, ctr=None):
    """
    :param t: context string
    :param lock: multiprocessing Lock
    :param ctr: context counter (for displaying progress in multiprocessing mode)
    :return: a dictionary consisting of 'context', 'clusters', 'resolved' where 'context' is the original text,
    and 'resolved' is the output of replacing coreferent entity 'clusters' with their head (canonical) term.
    """
    coref_obj = construct_coref(t)
    coref_obj['context'] = t
    with lock:
        ctr.value += 1
        if ctr.value % update_incr == 0:
            print('Processed {} contexts...'.format(ctr.value))

    return coref_obj


def resolve_corefs(dataset, dtype):
    """
    :param dataset: a subclass of preprocessing.DatasetBase
    :param dtype: one of 'mini', 'train', 'validation', 'test'
    :return: None
    Resolves coreferences for all contexts in dataset and saves to '../data/{dataset_name}/contexts_{dtype}.json'
    Saves as dictionary where keys are Dataset defined UUIDs for context passages and value is a dictionary consisting
    of 'context', 'clusters', 'resolved' where 'context' is the original text, and 'resolved' is the output of
    replacing coreferent entity 'clusters' with their head (canonical) term.
    """
    print('Loading {} set...'.format(dtype))
    keys, texts = dataset.get_context_kv_pairs(dtype)
    print('Processing {} contexts for {} set...'.format(len(keys), dtype))
    with Manager() as manager:
        p = Pool()
        lock = manager.Lock()
        ctr = manager.Value('i', 0)
        coref_outputs = list(p.map(partial(process_context, lock=lock, ctr=ctr), texts))
    output_dict = {}
    for k, v in zip(keys, coref_outputs):
        output_dict[k] = v
    out_fn = os.path.join('..', 'data', dataset.name, 'contexts_{}.json'.format(dtype))
    print('Saving {} contexts to {}...'.format(len(output_dict), out_fn))
    with open(out_fn, 'w') as fd:
        json.dump(output_dict, fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Coreference Resolution Preprocessing Script.')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    dataset = dataset_factory(args.dataset)

    device = 0 if torch.cuda.is_available() else -1
    update_incr = 10 if args.debug else 10000

    print('Loading Coref Pipeline')
    coref_nlp = spacy.load('en_core_web_lg')
    coref = neuralcoref.NeuralCoref(coref_nlp.vocab)
    coref_nlp.add_pipe(coref, name='neuralcoref')

    dtypes = ['mini'] if args.debug else ['train', 'test', 'validation']
    for dtype in dtypes:
        start_time = time()
        resolve_corefs(dataset, dtype)
        duration(start_time)
