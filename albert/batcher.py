# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:39:40 2020

@author: Mert Ketenci
"""

import torch
import random

class Batcher:
    def __init__(self, train_size, test_size, batch_size, device):
        
        self.train_size = train_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.device = device
        
        self.train_ids = list(range(train_size))
        self.test_ids = list(range(test_size))
        
        self.start_idx = 0
        self.end_idx = self.start_idx + self.batch_size
        
        random.shuffle(self.train_ids)
        
    def next(self, generator, is_train = True):
        
        if is_train:
            
            ids = self.train_ids[self.start_idx:self.end_idx]
            
            x_train = torch.Tensor(generator.get_x(ids, 'train')).long().to(self.device)
            y_train = torch.Tensor(generator.get_y(ids, 'train')).long().to(self.device)
            token_type = torch.Tensor(generator.get_type(ids, 'train')).long().to(self.device)
            
            self.start_idx = self.end_idx
            self.end_idx = min(self.train_size-1,self.end_idx + self.batch_size)
            
            return x_train, y_train, token_type
        
        else:
            
            ids = self.test_ids[self.start_idx:self.end_idx]
            
            x_test = torch.Tensor(generator.get_x(ids, 'test')).long().to(self.device)
            y_test = torch.Tensor(generator.get_y(ids, 'test')).long().to(self.device)
            token_type = torch.Tensor(generator.get_type(ids, 'test')).long().to(self.device)
            
            self.start_idx = self.end_idx
            self.end_idx = min(self.test_size-1,self.end_idx + self.batch_size)
            
            return x_test, y_test, token_type
        
    def reset(self):
        
        random.shuffle(self.train_ids)
        self.start_idx = 0
        self.end_idx = self.start_idx + self.batch_size
