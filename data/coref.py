from collections import defaultdict
from functools import partial
import json
from multiprocessing import Manager, Pool
import os
import re
import sys
from time import time

from allennlp.predictors.predictor import Predictor
import neuralcoref
from nlp import load_dataset
import spacy
import torch
from torch.utils.data import Subset

from utils import dict_to_lists, duration, extract_contexts


def remove_extra_space(str):
    return re.sub(r'\s+', ' ', str)


def construct_coref(spacy_doc):
    clusters = defaultdict(set)
    coref = spacy_doc._
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
    coref_obj = construct_coref(coref_nlp(t))
    coref_obj['context'] = t
    with lock:
        ctr.value += 1
        if ctr.value % update_incr == 0:
            print('Processed {} contexts...'.format(ctr.value))

    return coref_obj


def process_dataset(dataset, out_fn):
    contexts = extract_contexts(dataset)
    keys, texts = dict_to_lists(contexts)
    print('Processing {} contexts...'.format(len(contexts)))
    with Manager() as manager:
        p = Pool()
        lock = manager.Lock()
        ctr = manager.Value('i', 0)
        coref_outputs = list(p.map(partial(process_context, lock=lock, ctr=ctr), texts))
    output_dict = {}
    for k, v in zip(keys, coref_outputs):
        output_dict[k] = v
    print('Saving {} contexts to {}...'.format(len(output_dict), out_fn))
    with open(out_fn, 'w') as fd:
        json.dump(output_dict, fd)


if __name__ == '__main__':
    debug_mode = len(sys.argv) > 1 and sys.argv[1] == 'debug'
    device = 0 if torch.cuda.is_available() else -1
    update_incr = 10 if debug_mode else 10000

    print('Loading Dataset...')
    debug_data_fn = 'trivia_qa/train_mini.json'
    if debug_mode and os.path.exists(debug_data_fn):
        with open(debug_data_fn, 'r') as fd:
            dataset = json.load(fd)
    elif debug_mode and not os.path.exists(debug_data_fn):
        dataset = load_dataset('trivia_qa', 'rc')['train']
        dataset = [dict(e) for e in Subset(dataset, list(range(1, 11)))]
        with open(debug_data_fn, 'w') as fd:
            json.dump(dataset, fd)
    else:
        dataset = load_dataset('trivia_qa', 'rc')
    print('Loading Coref Pipeline')
    coref_nlp = spacy.load('en_core_web_lg')
    coref = neuralcoref.NeuralCoref(coref_nlp.vocab)
    coref_nlp.add_pipe(coref, name='neuralcoref')

    start_time = time()
    if debug_mode:
        out_fn = os.path.join('trivia_qa', 'contexts_debug.json')
        process_dataset(dataset, out_fn)
    else:
        dtypes = list(sorted(list(dataset.keys())))  # test, train, validation
        print(dtypes)
        for dtype in dtypes:
            print('Resolving coreferent expressions for {} set'.format(dtype))
            out_fn = os.path.join('trivia_qa', 'contexts_{}.json'.format(dtype))
            process_dataset(dataset[dtype], out_fn)

    duration(start_time)
