from functools import partial
from multiprocessing import Pool, Manager
import os
import pickle

from nlp import load_dataset
import spacy
import neuralcoref


def process_corref(example, ctr=None, lock=None):
    contexts = example['entity_pages']
    docs = list(map(lambda d: spacy_nlp(d)._, contexts['wiki_context']))
    with lock:
        ctr.value += 1
        if ctr.value % 1 == 0:
            print('Processed {} examples'.format(ctr.value))

    # Include question_id and doc titles just to make sure we serialize in same order (as FK into full dataset)
    return {
        'question_id': example['question_id'],
        'entity_pages': {
            'title': contexts['title'],
            'coref': docs
        }
    }


if __name__ == '__main__':
    print('Loading Dataset')
    dataset = load_dataset('trivia_qa', 'rc')
    print('Loading Coref Pipeline')
    spacy_nlp = spacy.load('en_core_web_lg')
    coref = neuralcoref.NeuralCoref(spacy_nlp.vocab)
    spacy_nlp.add_pipe(coref, name='neuralcoref')

    dtypes = list(dataset.keys())  # train, test, etc,
    for dtype in dtypes:
        dataset_type = dataset[dtype]
        n = len(dataset_type)
        print('Processing {} examples from {} set'.format(n, dtype))
        out_fn = os.path.join('trivia_qa', 'coref_{}.pk'.format(dtype))
        with Manager() as manager:
            p = Pool()
            lock = manager.Lock()
            ctr = manager.Value('i', 0)
            clusters = list(p.map(partial(process_corref, ctr=ctr, lock=lock), dataset_type))
            qids = set(list(map(lambda c: c['question_id'], clusters)))
        print('Unique QIDs={}'.format(len(qids)))
        print(len(clusters))

        with open(out_fn, 'wb') as fd:
            pickle.dump(clusters, fd)
        print('Saved pickled output to {}'.format(out_fn))
