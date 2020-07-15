# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:39:40 2020

@author: Mert Ketenci
"""

import torch
import random

class Batcher:
    def __init__(self, train_size, test_size, train_batch_size, test_batch_size, device):
        
        self.train_size = train_size
        self.test_size = test_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = device
        
        self.train_ids = list(range(train_size))
        self.test_ids = list(range(test_size))
        
        self.train_start_idx = 0
        self.train_end_idx = self.train_start_idx + self.train_batch_size
        
        self.test_start_idx = 0
        self.test_end_idx = self.test_start_idx + self.test_batch_size
        
        random.shuffle(self.train_ids)
        random.shuffle(self.test_ids)
        
    def next(self, generator, is_train = True):
        
        if is_train:
            
            ids = self.train_ids[self.train_start_idx:self.train_end_idx]
            
            x_train = torch.Tensor(generator.get_x(ids, 'train')).long().to(self.device)
            token_type = torch.Tensor(generator.get_type(ids, 'train')).long().to(self.device)
            y_train = torch.Tensor(generator.get_y(ids, 'train')).long().to(self.device)
            
            self.train_start_idx = self.train_end_idx
            self.train_end_idx = min(self.train_size-1,self.train_end_idx + self.train_batch_size)
            
            return x_train, y_train, token_type
        
        else:
            
            ids = self.test_ids[self.test_start_idx:self.test_end_idx]
            
            x_test = torch.Tensor(generator.get_x(ids, 'test')).long().to(self.device)
            token_type = torch.Tensor(generator.get_type(ids, 'test')).long().to(self.device)
            y_answer = generator.get_tokenized_answer_list(ids, 'test', True)
            y_context =  generator.get_tokenized_context_list(ids, 'test', True)
            
            self.test_start_idx = self.test_end_idx
            self.test_end_idx = min(self.test_size-1,self.test_end_idx + self.test_batch_size)
            
            return x_test, y_context, y_answer, token_type
        
    def reset(self):
        
        random.shuffle(self.train_ids)
        self.train_start_idx = 0
        self.train_end_idx = self.train_start_idx + self.train_batch_size

        random.shuffle(self.test_ids)
        self.test_start_idx = 0
        self.test_end_idx = self.test_start_idx + self.test_batch_size
