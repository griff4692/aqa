import pickle
import sys

from nlp import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == '__main__':
    debug_mode = len(sys.argv) > 1 and sys.argv[1] == 'debug'
    tfidf = TfidfVectorizer(stop_words='english', min_df=2)

    print('Loading dataset...')
    dataset = load_dataset('../data/trivia_qa', 'rc')['train']

    token_corpus = []
    n = len(dataset)
    print('Adding corpus documents.')
    for i, example in enumerate(dataset):
        token_corpus += example['entity_pages']['wiki_context']
        if (i + 1) % 10000 == 0:
            print('Added wiki docs from {} out of {} examples'.format(i + 1, n))
    print('Fitting TF-IDF vectorizer')
    tfidf.fit(token_corpus)

    out_fn = 'trivia_qa/tf_idf_vectorizer.pk'
    print('Saving vectorizer to {}'.format(out_fn))
    with open('../data/trivia_qa/tf_idf_vectorizer.pk', 'wb') as fd:
        pickle.dump(tfidf, fd)
