# -*- coding: utf-8 -*-
"""

Created on Sun Jul 12 11:26:03 2020

@author: Mert Ketenci

Generator object : preprocessor and the kernel of the batcher

Note : We are already padding the tokens (ex : set get_tokenized_context_list(decode = True) to see the padded input)
Therefore, we are not going to build the attention_mask variable
see : https://stackoverflow.com/questions/60397610/use-of-attention-mask-during-the-forward-pass-in-lm-finetuning

ToDos : 
    
    (A) Discuss how to approach large context size : 
    
    (1) Many of the context @ hotpot dataset does not contain the answer in it.
    Shall we still use them?
    (2) Let's use sequence window?
    
    (B) Discuss how to approach reasoning questions (yes / no)
    
    (1) Are we only going to use factoidal question / answer pairs?
    

"""

import argparse
from collections import defaultdict
from itertools import chain
import json
from operator import itemgetter
import glob, os
from pathlib import Path
import pickle
from random import random

import numpy as np
from transformers import AlbertTokenizer, logging
from tqdm import tqdm

from utils import *

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        
path = os.path.join(Path(os.path.dirname(__file__)))
os.chdir(path)

class Generator:
    def __init__(self, dataset, train_test_split, max_pos):
                
        if dataset == 'hotpot':
            self.dataset = dataset
        elif dataset == 'trivia':
            self.dataset = dataset
        else:
            sys.exit('dataset arg should be hotpot or trivia')
        
        self.train_test_split = train_test_split
        self.max_pos = max_pos
        
        self.n_example = defaultdict(list)

        self.tokenized_context_list = defaultdict(list)
        self.tokenized_question_list = defaultdict(list)
        self.tokenized_answer_list = defaultdict(list)
        
        self.input_array = defaultdict(list)
        self.token_type_array = defaultdict(list)
        self.output_array = defaultdict(list)        
    
    def preprocess(self, data):
          
        output_list = defaultdict(list)
        input_list = defaultdict(list)
        
        n_question = defaultdict(list)
        
        self.n_example['total'] = len(data)
        if self.dataset == 'hotpot':
            for datum in tqdm(data, total = self.n_example['total']):
                
                if random() > self.train_test_split:
                    dtype = 'train'
                else:
                    dtype = 'test'
                    
                contexts = datum['context']
                question = datum['question']                
                answer = datum['answer']
                
                tokenized_context = []
                for context in contexts:
                    context = ''.join(context[-1])
                    if find_sub_list(answer,context) != None:
                        current_tokenized_context = tokenizer.encode(context)
                        tokenized_context.append(current_tokenized_context[1:-1])         
                    else:
                        pass
                    
                tokenized_context = flat_integer_list(tokenized_context)
                
                tokenized_question = tokenizer.encode(question)
                tokenized_answer = tokenizer.encode(answer)
                answer_span = find_sub_list(tokenized_answer[1:-1], tokenized_context)
                if answer_span != None:
                    self.tokenized_context_list[dtype].append(tokenized_context)
                    self.tokenized_answer_list[dtype].append(tokenized_answer)
                    self.tokenized_question_list[dtype].append(tokenized_question)
                    output_list[dtype].append(answer_span)
            
            count = defaultdict(int)
            for dtype in ['train','test']:
                input_list[dtype] = [x + y + ['3'] for (x, y) in zip(self.tokenized_question_list[dtype], 
                                                                     self.tokenized_context_list[dtype])]
                
                n_question[dtype] = [len(x) for x in self.tokenized_question_list[dtype]]
                                                                                    
                count[dtype] = max([len(x) for x in input_list[dtype]])
                self.n_example[dtype] = len(input_list[dtype])
                
            for dtype in ['train','test']:
                self.input_array[dtype] = np.zeros((self.n_example[dtype], count[dtype]), dtype = np.int16)
                self.token_type_array[dtype] = np.zeros((self.n_example[dtype], count[dtype]), dtype = np.int8)
                for i in range(self.n_example[dtype]):
                    token_size = len(input_list[dtype][i])
                    self.input_array[dtype][i,:token_size] = input_list[dtype][i]
                    self.token_type_array[dtype][i,n_question[dtype][i]:] = 1
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
    
    def get_type(self, ids, dtype):
        
        """
        returns the type of phrase [0] * len(question tokens) + [1] * len(context tokens)
        """ 
        
        return self.token_type_array[dtype][ids]
    
    def get_size(self, dtype):
        
        """
        returns number of examples in training / test sets
        """ 
        
        return self.n_example[dtype]
       
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build the generator.')
    parser.add_argument('-dataset', default = 'hotpot', help = 'question answering dataset')
    parser.add_argument('-len_max_seq', default = 512, help = 'maximum sequence length')
    parser.add_argument('-train_test_split', default = 0.2, help = 'train / test split ratio')
    args = parser.parse_args()
    
    print('Loading dataset...')
    input_fn = '../data/hotpot_qa/hotpot_train_v1.1.json'
    with open(input_fn) as f:
        dataset = json.load(f)
        
    print('Building generator...\n')
    generator = Generator(args.dataset, args.train_test_split, args.len_max_seq)
    generator.preprocess(dataset)
    
    print('Saving...')
    output_fn = '../data/hotpot_qa/generator.pk'
    with open(output_fn, 'wb') as fd:
         pickle.dump(generator, fd, protocol=4)
    

