import argparse
import json
import pickle
import sys

from nlp import load_dataset
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from utils import clean_text

from tfidf import TfidfHommade

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Build tf-idf method.')
    parser.add_argument('-use_sklearn', default= False,
                        help='use scikit-learn or hommade vectorizer')
    parser.add_argument('-dataset', default= 'hotpot',
                        help='qa dataset')
    args = parser.parse_args()
    
    if args.dataset == 'hotpot':
        source = 'hotpot_qa'
    elif args.dataset == 'trivia':
        source = 'trivia_qa'
    else:
        sys.exit('dataset should be hotpot or trivia')
        
    print('Loading dataset...')
    input_fn = './hotpot_qa/hotpot_train_v1.1.json'
    with open(input_fn) as f:
        dataset = json.load(f)
    
    token_corpus = []
    total = len(dataset)
    print('Adding corpus documents...\n')
    for examples in tqdm(dataset, total = total):
        for example in examples['context']:
            token_corpus += [clean_text(' '.join(example[1]))]
                
    if args.use_sklearn:
        tfidf = TfidfVectorizer(stop_words='english', min_df=2)
        print('Fitting scikit-learn TF-IDF vectorizer')
        tfidf.fit(token_corpus)
        out_fn = source + '/tf_idf_vectorizer_sklearn.pk'
        print('Saving vectorizer to {}'.format(out_fn))
        with open(out_fn, 'wb') as fd:
            pickle.dump(tfidf, fd)
    else:
        tfidf = TfidfHommade()
        print('Fitting homemade TF-IDF vectorizer')
        tfidf.fit(token_corpus)
        out_fn = source + '/tf_idf_vectorizer_hommade.pk'
        print('Saving vectorizer to {}'.format(out_fn))
        with open(out_fn, 'wb') as fd:
            pickle.dump(tfidf, fd)

