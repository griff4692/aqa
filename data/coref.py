from collections import defaultdict
import os
import json
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


def process_contexts(contexts, is_debug=True):
    debug_str = '_mini' if is_debug else ''
    out_fn = os.path.join('trivia_qa', 'contexts{}.json'.format(debug_str))
    outputs = {}

    for k, t in contexts.items():
        coref_obj = construct_coref(coref_nlp(t))
        coref_obj['context'] = t
        outputs[k] = coref_obj

    print('Saving {} contexts to {}'.format(len(outputs), out_fn))
    with open(out_fn, 'w') as fd:
        json.dump(outputs, fd)


if __name__ == '__main__':
    debug_mode = len(sys.argv) > 1 and sys.argv[1] == 'debug'
    device = 0 if torch.cuda.is_available() else -1

    print('Loading Dataset')
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
        contexts = extract_contexts(dataset)
        print('Processing {} contexts in debug mode'.format(len(contexts)))
        process_contexts(contexts)
    else:
        dtypes = list(dataset.keys())  # train, test, validation
        for dtype in dtypes:
            d = dataset[dtype]
            contexts = extract_contexts(d)
            n = len(d)
            print('Processing {} contexts from {}'.format(len(contexts), dtype))
            process_contexts(contexts)

    duration(start_time)
