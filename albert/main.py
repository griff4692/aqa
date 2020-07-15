# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:35:22 2020

@author: Mert Ketenci
"""

import argparse
import numpy as np
from pathlib import Path
import pickle
import sys,os
import torch
from tqdm import tqdm

from generator import Generator
from batcher import Batcher
from model import AlbertQAModel

path = os.path.join(Path(os.path.dirname(__file__)).parent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script to train alber qa model')
    
    parser.add_argument('-n_epoch',default = 600)    
    parser.add_argument('-train_batch_size',default = 2)
    parser.add_argument('-test_batch_size',default = 2)
    parser.add_argument('-lr',default = 1e-10)
    parser.add_argument('-device',default = 'cpu')
    
    args = parser.parse_args()
        
    print('Loading the generator...')
    with open(path + '/data/hotpot_qa/generator.pk', 'rb') as fd:
        generator = pickle.load(fd)
        
    train_size = generator.get_size('train')
    test_size = generator.get_size('test')
    max_pos = generator.get_pos()
    
    
    batcher = Batcher(train_size, test_size, args.train_batch_size, args.test_batch_size, args.device)
    
    model = AlbertQAModel(max_pos).to(args.device)
    
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr = args.lr)
    
    train_epoch_len = train_size // args.train_batch_size + 1
    test_epoch_len = test_size // args.test_batch_size + 1
    
    train_loss = []
    test_loss = []
    exact_match_list = []
    
    model.train()
    batcher.reset()
    beta = 0
    
    for e in range(args.n_epoch):
        for i in tqdm(range(train_epoch_len), position=0, leave=True):
             
            optimizer.zero_grad()
             
            x_train, y_train, token_type = batcher.next(generator, is_train = True)
            loss = model.forward(x_train = x_train, y_train = y_train, 
                                 token_type = token_type, is_train = True)

            optimizer.step()
        
        for i in tqdm(range(test_epoch_len)):
        
            x_test, y_context, y_answer, token_type = batcher.next(generator, is_train = False)
            exact_match = model.eval().forward(x_test = x_test, y_context = y_context, 
                                        y_answer = y_answer, token_type = token_type, is_train = False)

            exact_match_list.append(exact_match)
        
        epoch.exact_match_list.append(np.mean(exact_match_list))
        
        model.train()
        batcher.reset()
        print("Exact match is : {}".format(exact_match_list[-1]))
        