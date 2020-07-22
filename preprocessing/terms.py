import pickle
import os
from string import punctuation
import sys

import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from tqdm import tqdm

from dataset_base import dataset_factory
from utils import duration, remove_extra_space


def tokenize(str):
    toks = [token.text.lower().strip(punctuation).strip() for token in spacy_nlp(str)]
    return [tok for tok in toks if len(tok) > 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Coreference Resolution Preprocessing Script.')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    args = parser.parse_args()

    dataset = dataset_factory(args.dataset)
    tfidf = TfidfVectorizer(stop_words='english', min_df=1)
    spacy_nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner', 'tagger'])

    print('Loading train set...')
    _, train_contexts = dataset.get_context_kv_pairs('train')
    print('Loading validation set...')
    _, val_contexts = dataset.get_context_kv_pairs('validation')
    print('Loading test set...')
    _, test_contexts = dataset.get_context_kv_pairs('test')
    corpus = []
    contexts = train_contexts + val_contexts + test_contexts
    n = len(contexts)
    print('Adding corpus documents.')
    for i in tqdm(range(n)):
        corpus.append(' '.join(tokenize(contexts[i])))
        if (i + 1) % 100000 == 0 or (i + 1) == n:
            print('Added wiki docs from {} out of {} examples'.format(i + 1, n))
    print('Fitting TF-IDF vectorizer')
    tfidf.fit(corpus)

    out_fn = os.path.join('..', 'data', dataset.name, 'tf_idf_vectorizer.pk')
    print('Saving vectorizer to {}'.format(out_fn))
    with open(out_fn, 'wb') as fd:
        pickle.dump(tfidf, fd)
