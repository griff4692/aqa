# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:51:47 2020

@author: Mert Ketenci
"""

from collections import defaultdict
from collections import Counter
import json
import time

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from utils import overlap, clean_text


class TfidfHommade:
    
    def __init__(self):
        self.idf = defaultdict(int)
        
    def fit(self, documents):
        self.log_num_documents = np.log(len(documents) + 1)
        for document in documents:
            document = set(document.split(' '))
            for word in document:
                self.idf[word] += 1
        for key in self.idf.keys():
            self.idf[key] = self.idf[key]
        
    def similarity(self, x, y):
        x = x.lower().split(' ')
        y = y.lower().split(' ')
        
        set_x = set(x)
        set_y = set(y)
        
        intersection = set_x & set_y
        tf_x_intersection = self.calc_tf(x, intersection)
        tf_y_intersection = self.calc_tf(y, intersection)
        
        idf_intersection = self.calc_idf(intersection)
        
        tf_idf_x_intersection = self.calc_tf_idf(tf_x_intersection, idf_intersection) 
        tf_idf_y_intersection = self.calc_tf_idf(tf_y_intersection, idf_intersection)
        
        tf_x = self.calc_tf(x, set_x)
        tf_y = self.calc_tf(y, set_y)
        
        idf_x = self.calc_idf(set_x)
        idf_y = self.calc_idf(set_y)
        
        tf_idf_x = self.calc_tf_idf(tf_x, idf_x)
        tf_idf_y = self.calc_tf_idf(tf_y, idf_y)
        
        tf_idf_x_intersection /= np.sqrt(np.dot(tf_idf_x, tf_idf_x))
        tf_idf_y_intersection /= np.sqrt(np.dot(tf_idf_y, tf_idf_y))
        return self.cosine_sim(tf_idf_x_intersection, tf_idf_y_intersection)
                                
        
    def calc_tf(self, x, unique):
        tf = Counter(x)
        return np.asarray([tf[word] for word in unique]) / len(x)
    
    def calc_idf(self, words):        
        return self.log_num_documents - np.log([self.idf[word] + 1 for word in words]) + 1
    
    def calc_tf_idf(self, tf, idf):
        return tf * idf
    
    def cosine_sim(self, tf_idf_x_intersection, tf_idf_y_intersection):
        cosine = np.dot(tf_idf_x_intersection, tf_idf_y_intersection)
        return cosine
   
if __name__ == '__main__':
    
    print('Loading dataset...')
    input_fn = './trivia_qa/train_mini.json'
    with open(input_fn) as f:
        dataset = json.load(f)
    
    token_corpus = []
    n = len(dataset)
    print('Adding corpus documents.')
    for i, example in enumerate(dataset):
        token_corpus += example['entity_pages']['wiki_context']
        if (i + 1) % 10000 == 0:
            print('Added wiki docs from {} out of {} examples'.format(i + 1, n))
            
    documents = [clean_text(text) for text in token_corpus]
    
    a = ['england is in uk', 'uk includes england']
    
    print('Fitting TF-IDF vectorizer')
    tfidf = tf_idf()
    tfidf.fit(documents)
    
    vectorizer = TfidfVectorizer()
    tf_idf_sk = vectorizer.fit(documents)
    
    ourtf = []
    theirtf = []
    
    for i in tqdm(range(100)):
        
        t0 = time.time()
        tfidf.similarity(a[0], a[1])
        t1 = time.time()
        ourtf.append(t1-t0)
        
        t0 = time.time()
        overlap(a[0], a[1], tf_idf_sk)
        t1 = time.time()
        theirtf.append(t1-t0)
    
    print('Our tfidf is {} times faster'.format(np.mean(theirtf) / np.mean(ourtf)))