# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:39:40 2020

@author: Mert Ketenci
"""

class Batcher:
    def __init__(self, train_size, test_size, batch_size):
        
        self.train_size = train_size
        self.test_size = test_size
        
        self.train_ids = list(range(train_size))
        self.test_ids = list(range(test_size))
        
        self.start_idx = 0
        self.end_idx = self.start_idx + self.batch_size
        
        random.shuffle(self.train_ids)
        
    def next(generator, is_train):
        
        if is_train:
            
            ids = self.train_ids[self.start_idx:self.end_idx]
            
            x_train = generator.get_x(ids, 'train')
            y_train = generator.get_y(ids, 'train')
            
            self.start_idx = self.end_idx
            self.end_idx = min(self.train_size-1,self.end_idx + self.batch_size)
            
            return x_train, y_train
        
        else:
            
            ids = self.test_ids[self.start_idx:self.end_idx]
            
            x_test = generator.get_x(ids, 'test')
            y_test = generator.get_y(ids, 'test')
            
            self.start_idx = self.end_idx
            self.end_idx = min(self.test_size-1,self.end_idx + self.batch_size)
            
            return x_test, y_test
        
    def reset(self):
        
        random.shuffle(self.train_ids)
        self.start_idx = 0
        self.end_idx = self.start_idx + self.batch_size
