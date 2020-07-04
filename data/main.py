from functools import partial
import json
from multiprocessing import Pool, Manager
import os
import sys

import neuralcoref
from nlp import load_dataset
import spacy
from torch.utils.data import Subset


def process_corref(example, ctr=None, lock=None):
    contexts = example['entity_pages']
    docs = list(map(lambda d: spacy_nlp(d).to_json(), contexts['wiki_context']))
    if len(docs) > 0:
        print(docs[0])
    with lock:
        ctr.value += 1
        if ctr.value % 1 == 0:
            print('Processed {} examples'.format(ctr.value))

    # Include question_id and doc titles just to make sure we serialize in same order (as FK into full dataset)
    return {
        'question_id': example['question_id'],
        'entity_pages': {
            'title': contexts['title'],
            'spacy_doc': docs
        }
    }


if __name__ == '__main__':
    debug_mode = len(sys.argv) > 1 and sys.argv[1] == 'debug'

    print('Loading Dataset')
    dataset = load_dataset('trivia_qa', 'rc')
    print('Loading Coref Pipeline')
    spacy_nlp = spacy.load('en_core_web_lg')
    coref = neuralcoref.NeuralCoref(spacy_nlp.vocab)
    spacy_nlp.add_pipe(coref, name='neuralcoref')

    dtypes = list(dataset.keys())  # train, test, etc,
    for dtype in dtypes:
        d = dataset[dtype]
        if debug_mode:
            d = Subset(d, list(range(2)))
        n = len(d)
        print('Processing {} examples from {} set'.format(n, dtype))
        out_fn = os.path.join('trivia_qa', 'coref_{}.json'.format(dtype))
        with Manager() as manager:
            p = Pool()
            lock = manager.Lock()
            ctr = manager.Value('i', 0)
            clusters = list(p.map(partial(process_corref, ctr=ctr, lock=lock), d))
            qids = set(list(map(lambda c: c['question_id'], clusters)))
        print('Unique QIDs={}'.format(len(qids)))
        print(len(clusters))

        with open(out_fn, 'w') as fd:
            json.dump(clusters, fd)
        print('Saved pickled output to {}'.format(out_fn))
