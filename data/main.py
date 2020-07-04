from functools import partial
from multiprocessing import Pool, Manager
import os
import pickle
import sys

from allennlp.predictors.predictor import Predictor
from nlp import load_dataset
from torch.utils.data import Subset


def construct_graph(example, ctr=None, lock=None):
    contexts = list(map(lambda x: {'sentence': x}, example['entity_pages']['wiki_context']))
    print(contexts)
    coref_pred = predictor.predict_batch_json(contexts)
    with lock:
        ctr.value += 1
        if ctr.value % 1 == 0:
            print('Processed {} examples'.format(ctr.value))

    print(coref_pred)

    # Include question_id and doc titles just to make sure we serialize in same order (as FK into full dataset)
    return {
        'question_id': example['question_id'],
        'entity_pages': {
            'title': example['entity_pages']['title'],
            'coref': coref_pred
        }
    }


if __name__ == '__main__':
    debug_mode = len(sys.argv) > 1 and sys.argv[1] == 'debug'

    print('Loading Dataset')
    dataset = load_dataset('trivia_qa', 'rc')
    print('Loading Coref Pipeline')
    predictor = Predictor.from_path(
        'https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz')

    dtypes = list(dataset.keys())  # train, test, etc,
    for dtype in dtypes:
        d = dataset[dtype]
        if debug_mode:
            d = Subset(d, list(range(1, 3)))
        n = len(d)
        print('Processing {} examples from {} set'.format(n, dtype))
        out_fn = os.path.join('trivia_qa', 'coref_{}.pk'.format(dtype))
        with Manager() as manager:
            p = Pool()
            lock = manager.Lock()
            ctr = manager.Value('i', 0)
            clusters = list(map(partial(construct_graph, ctr=ctr, lock=lock), d))
            qids = set(list(map(lambda c: c['question_id'], clusters)))
        print('Unique QIDs={}'.format(len(qids)))
        print(len(clusters))
        with open(out_fn, 'wb') as fd:
            pickle.dump(clusters, fd)
        print('Saved pickled output to {}'.format(out_fn))
