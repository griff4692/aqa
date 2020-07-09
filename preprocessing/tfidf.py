# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:51:47 2020

@author: Mert Ketenci
"""

from collections import defaultdict
from collections import Counter
import time


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import string


def clean_text(text, ignore_paranthesis = True):
    if text!=None:
        if ignore_paranthesis:
            text = re.sub("[\(\[].*?[\)\]]", "", text)
            
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.lower()
    

class tf_idf:
    
    def __init__(self):
        self.idf = defaultdict(int)
        
    def fit(self, documents):
        num_documents = len(documents)
        for document in documents:
            document = set(document.split(' '))
            for word in document:
                self.idf[word] += 1
        for key in self.idf.keys():
            self.idf[key] = num_documents / self.idf[key]

    def similarity(self, x, y):
        x = x.split(' ')
        y = y.split(' ')
        
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
        
        return self.cosine_sim(tf_idf_x_intersection, 
                               tf_idf_y_intersection, tf_idf_x, tf_idf_y)
        
    def calc_tf(self, x, intersection):
        tf = Counter(x)
        return np.asarray([tf[word] for word in intersection]) / len(x)
    
    def calc_idf(self, intersection):        
        return np.log([max(1, self.idf[word]) for word in intersection])
    
    def calc_tf_idf(self, tf, idf):
        return tf * idf
    
    def cosine_sim(self, tf_idf_x_intersection, tf_idf_y_intersection, tf_idf_x, tf_idf_y):
        cosine = np.dot(tf_idf_x_intersection, tf_idf_y_intersection)
        mag_x = np.dot(tf_idf_x, tf_idf_x)
        mag_y = np.dot(tf_idf_y, tf_idf_y)
        return cosine/(np.sqrt(mag_x * mag_y) + 1e-40)
   

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
        
token_corpus = [clean_text(text) for text in token_corpus]

print('Fitting TF-IDF vectorizer')
tfidf = tf_idf()
tfidf.fit(token_corpus)

vectorizer = TfidfVectorizer()
tf_idf_sk = vectorizer.fit(token_corpus)

a = ['See me walking down Fifth Avenue', 'A walking cane here at my side']
a  = [x.lower() for x in a]
 
t0 = time.time()
print(tfidf.similarity(a[0], a[1]))
t1 = time.time()
print(t1-t0)


def overlap(a, b, tf_idf_sk):
    feats = tf_idf_sk.transform([a.lower(), b.lower()])
    return (feats[0, :] * feats[1, :].T).todense()[0, 0]


t0 = time.time()
print(overlap(a[0], a[1], tf_idf_sk))
t1 = time.time()
print(t1-t0)
