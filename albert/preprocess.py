# -*- coding: utf-8 -*-
"""

Created on Sun Jul 12 11:26:03 2020

@author: Mert Ketenci

Batcher and preprocess in one method

"""
import argparse
from collections import defaultdict
from itertools import chain
import json
from operator import itemgetter
from random import random

import numpy as np
from transformers import AlbertTokenizer, AlbertConfig, AlbertForMaskedLM

from utils import *

configuration = AlbertConfig()
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        

class Generator:
    def __init__(self, dataset, train_test_split):
                
        if dataset == 'hotpot':
            self.dataset = dataset
        elif dataset == 'trivia':
            self.dataset = dataset
        else:
            sys.exit('dataset arg should be hotpot or trivia')
        
        self.train_test_split = train_test_split
        self.tokenized_context_list = defaultdict(list)
        self.tokenized_question_list = defaultdict(list)
        self.tokenized_answer_list = defaultdict(list)
        self.input_array = defaultdict(list)
        self.output_array = defaultdict(list)        
    
    def preprocess(self, data):
          
        output_list = defaultdict(list)
        input_list = defaultdict(list)
        n_example = defaultdict(list)
        
        n_example['total'] = len(data)
        
        if self.dataset == 'hotpot':
            for datum in data:
                
                if random() > self.train_test_split:
                    dtype = 'train'
                else:
                    dtype = 'test'
                    
                contexts = datum['context']
                question = datum['question']                
                answer = datum['answer']
                
                current_tokenized_context_list = []
                for context in contexts:
                    current_context = ''.join(context[-1])
                    current_tokenized_context = tokenizer.encode(current_context)
                    current_tokenized_context_list.append(current_tokenized_context[1:-1])           
                
                current_tokenized_context_list = flat_integer_list(current_tokenized_context_list)
                current_tokenized_question = tokenizer.encode(question)
                current_tokenized_answer = tokenizer.encode(answer)
                
                self.tokenized_context_list[dtype].append(current_tokenized_context_list)
                self.tokenized_answer_list[dtype].append(current_tokenized_answer)
                self.tokenized_question_list[dtype].append(current_tokenized_question)
                
                output_list[dtype].append(find_sub_list(current_tokenized_answer[1:-1], current_tokenized_context_list))
            
            count = defaultdict(int)
            for dtype in ['train','test']:
                input_list[dtype] = [x + y + ['3'] for (x, y) in zip(self.tokenized_question_list[dtype], 
                                                                     self.tokenized_context_list[dtype])]
                count[dtype] = max([len(x) for x in input_list[dtype]])
                n_example[dtype] = len(input_list[dtype])
                
            for dtype in ['train','test']:
                self.input_array[dtype] = np.zeros((n_example[dtype] + 1, count[dtype]), dtype = np.int16)
                for i in range(n_example[dtype]):
                    token_size = len(input_list[dtype][i])
                    self.input_array[dtype][i,:token_size] = input_list[dtype][i]
            
                self.output_array[dtype] = np.asarray(output_list[dtype])
            
    def get_tokenized_context_list(self, ids, dtype, decode = False):
        
        """
        returns [CLS] context tokens [SEP] for training set if dtype = train and decode = False
        returns [CLS] context phrase [SEP] for training set if dtype = train and decode = True
        """
        
        tokenized_context_list = itemgetter(*ids)([['2'] + x + ['3'] for x in self.tokenized_context_list[dtype]])
        if decode:
            return [tokenizer.decode(x) for x in tokenized_context_list]
        else:
            return tokenized_context_list
    
    def get_tokenized_question_list(self, ids, dtype, decode = False):
        
        """
        returns [CLS] question tokens [SEP] for training set if dtype = train and decode = False
        returns [CLS] question phrase [SEP] for training set if dtype = train and decode = True
        """
        
        tokenized_question_list = itemgetter(*ids)(self.tokenized_question_list[dtype])
        if decode:
            return [tokenizer.decode(x) for x in tokenized_question_list]
        else:
            return tokenized_question_list
    
    def get_tokenized_answer_list(self, ids, dtype, decode = False):
        
        """
        returns [CLS] answer tokens [SEP] for training set if dtype = train and decode = False
        returns [CLS] answer phrase [SEP] for training set if dtype = train and decode = True
        """
        
        tokenized_answer_list = itemgetter(*ids)(self.tokenized_answer_list[dtype])
        if decode:
            return [tokenizer.decode(x) for x in tokenized_answer_list]
        else:
            return tokenized_answer_list
    
    def get_x(self, ids, dtype, decode = False):
        
        """
        returns [CLS] question tokens [SEP] context tokens [CLS] for training set if dtype = train and decode = False
        returns [CLS] question phrase [SEP] context phrase [CLS] for training set if dtype = train and decode = True
        """
        
        input_array = self.input_array[dtype][ids]
        if decode: 
            return [tokenizer.decode(x) for x in input_array]
        else:
            return input_array
    
    def get_y(self, ids, dtype):
        
        """
        returns <BEG> <END> of answer (span) with respect to context for training set if dtype = train
        """        

        output_array = self.output_array[dtype][ids]
        return output_array
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build the graph.')
    parser.add_argument('-dataset', default = 'hotpot', help = 'qa dataset')
    parser.add_argument('-train_test_split', default = 0.2, help = 'qa dataset')
    args = parser.parse_args()
    
    print('Loading dataset...')
    input_fn = '../data/hotpot_qa/hotpot_train_v1.1.json'
    with open(input_fn) as f:
        dataset = json.load(f)     
    
    generator = Generator(args.dataset, args.train_test_split)
    generator.preprocess(dataset)
    

