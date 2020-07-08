# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:12:29 2020

@author: Mert Ketenci
"""

import json
import pickle

from kg import _node_match
from utils import _extract_contexts

if __name__ == '__main__':
    
    print('Loading tf-idf vectorizer...')
    with open('trivia_qa/tf_idf_vectorizer.pk', 'rb') as fd:
        tf_idf = pickle.load(fd)

    print('Loading context...')
    with open('trivia_qa/contexts_ie_debug.json', 'rb') as fd:
        contexts_ie_debug = json.load(fd)
        
    print('Loading data...')
    with open('trivia_qa/train_mini.json', 'rb') as fd:
        train_mini = json.load(fd)
        
    graph = Graph()
    
    for key in contexts_ie_debug.keys():
        for ie_tup in contexts_ie_debug[key]['ie_tups']:
            node_1, edge, node_2 = ie_tup
            resolve_node_edge(node_1, node_2, edge, graph)