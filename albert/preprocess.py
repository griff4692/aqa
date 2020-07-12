# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:26:03 2020

@author: Mert Ketenci
"""
import argparse
import json

import numpy as np
from transformers import AlbertTokenizer, AlbertConfig, AlbertForMaskedLM

configuration = AlbertConfig()
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')


def flat_integer_list(nested_integer_list):
    x = []
    for i in nested_integer_list:
        x = x + i
    return x


def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1


class DataPreprocessor:
    def __init__(self, dataset, pretraining_input_size = 30000):
                
        if dataset == 'hotpot':
            self.dataset = dataset
        elif dataset == 'trivia':
            self.dataset = dataset
        else:
            sys.exit('dataset arg should be hotpot or trivia')
            
        self.tokenized_context_list = []
        self.tokenized_question_list = []
        self.tokenized_answer_list = []
        self.mask_attention_list = []
        self.pretraining_input_size = pretraining_input_size
        
    def preprocess(self, train_data):
        
        n_examples = len(train_data)
        
        output_list = []
        
        if self.dataset == 'hotpot':
            for train_datum in train_data:
                
                contexts = train_datum['context']
                
                question = train_datum['question']
                current_tokenized_question = tokenizer.encode(question)
                
                answer = train_datum['answer']
                current_tokenized_answer = tokenizer.encode(answer)
                
                context_list = []
                allowable_context_length = self.pretraining_input_size - len(current_tokenized_question[1:]) - 1
                
                current_tokenized_context_list = []
                current_context_length = 0
                for context in contexts:
                    current_context = ''.join(context[-1])
                    current_tokenized_context = tokenizer.encode(current_context)
                    current_context_length += len(current_tokenized_context)
                    # if current_context_length <= allowable_context_length:
                    current_tokenized_context_list.append(current_tokenized_context[1:-1])
                    # else:
                    #     break
                
                
                current_tokenized_context_list = flat_integer_list(current_tokenized_context_list)
                
                self.tokenized_context_list.append(current_tokenized_context_list)
                self.tokenized_answer_list.append(current_tokenized_answer)
                self.tokenized_question_list.append(current_tokenized_question)
                
                output_list.append(find_sub_list(current_tokenized_answer[1:-1], current_tokenized_context_list))
                
            input_list = [x + y + ['3'] for (x, y) in zip(self.tokenized_question_list, self.tokenized_context_list)]
                        
            self.input_array = np.zeros((n_examples, self.pretraining_input_size), dtype = np.int16)
            for i in range(n_examples):
                token_size = len(input_list[i])
                self.input_array[i,:token_size] = input_list[i]
                
            self.output_array = np.asarray(output_list)
            
    def get_tokenized_context_list(self):
        return [['2'] + x + ['3'] for x in  self.tokenized_context_list]
    
    def get_tokenized_question_list(self):
        return self.tokenized_question_list
    
    def get_tokenized_answer_list(self):
        return self.tokenized_answer_list
    
    def get_mask_attention_list(self):
        return self.mask_attention_list
    
    def get_x_train(self, ids):
        return self.input_array[ids]
    
    def get_y_train(self, ids):
        return self.output_array[ids]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build the graph.')
    parser.add_argument('-dataset', default= 'hotpot', help='qa dataset')
    args = parser.parse_args()
    
    print('Loading dataset...')
    input_fn = '../data/hotpot_qa/hotpot_train_v1.1.json'
    with open(input_fn) as f:
        dataset = json.load(f)     
    
    training_preprocessor = DataPreprocessor(args.dataset)
    training_preprocessor.preprocess(dataset[:20])
    

