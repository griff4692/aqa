from collections import defaultdict
from functools import partial
import json
from multiprocessing import Manager, Pool
import os
import re
from time import time

import argparse
from bs4 import BeautifulSoup
import neuralcoref
import spacy
import torch

from dataset_base import dataset_factory
from utils import duration, remove_extra_space


def clean(str):
    str = BeautifulSoup(str, 'html.parser').get_text(strip=True)
    str = str.replace('–', '-')
    str = re.sub(r'[©®™]', ' ', str)
    str = re.sub(r'\s+', ' ', str)
    return str.strip()


def construct_coref(t):
    """
    :param t: string or list of strings
    :return: a dictionary consisting of 'clusters', 'resolved' where 'resolved' is the output of replacing coreferent
    entity 'clusters' with their head (canonical) term.
    """
    coref = coref_nlp(clean(t))._
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


def collect(out_dir):
    existing_fns = os.listdir(out_dir)
    output = {}
    for fn in existing_fns:
        with open(os.path.join(out_dir, fn), 'r') as fd:
            output.update(json.load(fd))
    return output


def dump_chunk(dict, out_dir):
    existing_fns = os.listdir(out_dir)
    if len(existing_fns) == 0:
        out_fn = os.path.join(out_dir, '0.json')
    else:
        nums = [int(x.split('.')[0]) for x in existing_fns]
        out_fn = os.path.join(out_dir, '{}.json'.format(max(nums) + 1))

    n = len(dict)
    print('Dumping {} examples to {}'.format(n, out_fn))
    with open(out_fn, 'w') as fd:
        json.dump(dict.copy(), fd)
    dict.clear()
    print('Now has {} items. Time to get more!'.format(len(dict)))


def process_context(input, lock=None, ctr=None, out_dir=None, outputs=None):
    """
    :param t: context string
    :param lock: multiprocessing Lock
    :param ctr: context counter (for displaying progress in multiprocessing mode)
    :return: a dictionary consisting of 'context', 'clusters', 'resolved' where 'context' is the original text,
    and 'resolved' is the output of replacing coreferent entity 'clusters' with their head (canonical) term.
    """
    k, t = input
    coref_obj = construct_coref(t)
    coref_obj['context'] = t
    with lock:
        ctr.value += 1
        outputs[k] = coref_obj

        if len(outputs) == 1000:
            dump_chunk(outputs, out_dir)

        if ctr.value % update_incr == 0:
            print('Processed {} contexts...'.format(ctr.value))

    return coref_obj


def resolve_corefs(dataset, dtype, out_dir, preexisting_keys):
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
    keys, texts = dataset.get_context_kv_pairs(dtype, skip_keys=preexisting_keys)
    print('Processing {} contexts for {} set...'.format(len(keys), dtype))
    with Manager() as manager:
        p = Pool(processes=10)
        lock = manager.Lock()
        ctr = manager.Value('i', 0)
        outputs = manager.dict()
        p.map(partial(process_context, lock=lock, ctr=ctr, outputs=outputs, out_dir=out_dir), zip(keys, texts))

        if len(outputs) > 0:
            dump_chunk(outputs, out_dir)


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
        final_out_dir = os.path.join('..', 'data', dataset.name)
        out_chunk_dir = os.path.join('..', 'data', dataset.name, 'chunks', dtype)
        if not os.path.exists(final_out_dir):
            print('Creating directory at {}'.format(final_out_dir))
            os.mkdir(final_out_dir)
        if not os.path.exists(out_chunk_dir):
            print('Creating directory at {}'.format(out_chunk_dir))
            os.mkdir(out_chunk_dir)
        preexisting = collect(out_chunk_dir)
        prev_n = len(preexisting)
        print('We\'ve already preprocessed {} examples.  Skipping them this time.'.format(prev_n))
        preexisting_keys = list(preexisting.keys())
        resolve_corefs(dataset, dtype, out_chunk_dir, preexisting_keys)
        collected_obj = collect(out_chunk_dir)
        n = len(collected_obj)
        final_out_fn = os.path.join(final_out_dir, 'contexts_{}.json'.format(dtype))
        print('Saving total {} examples to {}'.format(n, final_out_fn))
        with open(final_out_fn, 'w') as fd:
            json.dump(collected_obj, fd)
        duration(start_time)
