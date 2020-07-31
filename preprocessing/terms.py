from collections import Counter
import pickle
import os
from string import punctuation
import sys

import argparse
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from tqdm import tqdm

from dataset_base import dataset_factory


STOPWORDS = set(stopwords.words('english'))


def tokenize(str):
    toks = [token.text.lower().strip(punctuation).strip() for token in spacy_nlp(str)]
    return [tok for tok in toks if len(tok) > 0]


def get_default(idf):
    return max(idf.idf_)


def get_idf(x, idf, default=11.0):
    if x in idf.vocabulary_:
        return idf.idf_[idf.vocabulary_[x]]
    return default


def idf_mass(counts, idf, default=11.0):
    x = 0.0
    for k, v in counts.items():
        x += get_idf(k, idf, default=default) * v
    return x


def fast_idf(a, b, idf, default=11.0):
    a = [x.lower() for x in a]
    b = [x.lower() for x in b]
    a = list(filter(lambda x: x not in idf.stop_words, a))
    b = list(filter(lambda x: x not in idf.stop_words, b))
    a_counts = Counter(a)
    b_counts = Counter(b)
    a_idf = idf_mass(a_counts, idf, default=default)
    b_idf = idf_mass(b_counts, idf, default=default)

    c_idf = 0.0
    for k, v in a_counts.items():
        idf_val = get_idf(k, idf, default=default)
        min_coef = min(v, b_counts.get(k, 0.0))
        c_idf += min_coef * idf_val

    return a_idf, b_idf, c_idf


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Coreference Resolution Preprocessing Script.')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    args = parser.parse_args()

    dataset = dataset_factory(args.dataset)
    tfidf = TfidfVectorizer(stop_words=STOPWORDS, min_df=1)
    spacy_nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner', 'tagger'])

    print('Loading train set...')
    _, train_contexts = dataset.get_context_kv_pairs('train')
    print('Loading validation set...')
    _, val_contexts = dataset.get_context_kv_pairs('validation')

    if args.dataset == 'squad':
        test_contexts = []
    else:
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
