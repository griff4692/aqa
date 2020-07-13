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
    parser = argparse.ArgumentParser('main script to train alber qa model')
    
    parser.add_argument('-n_epoch',default = 600)    
    parser.add_argument('-batch_size',default = 1)
    parser.add_argument('-lr',default = 1e-10)
    parser.add_argument('-device',default = 'cuda')
    
    args = parser.parse_args()
        
    print('Loading the generator...')
    with open(path + '/data/hotpot_qa/generator.pk', 'rb') as fd:
        generator = pickle.load(fd)
        
    train_size = generator.get_size('train')
    test_size = generator.get_size('test')
    
    batcher = Batcher(train_size, test_size, args.batch_size, args.device)
    
    model = AlbertQAModel().to(args.device)
    
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr = args.lr)
    
    epoch_len = int(train_size/args.batch_size) + 1

    train_loss = []
    test_loss = []
    
    model.train()
    batcher.reset()
    beta = 0
    
    for e in range(args.n_epoch):
        for i in tqdm(range(epoch_len), position=0, leave=True):
             
            optimizer.zero_grad()
             
            x_train, y_train, token_type = batcher.next(generator, is_train = True)
             
            loss_start, loss_end = model.forward(x_train, y_train, token_type, is_train = True)
             
            loss_start.backward()
            loss_end.backward()
            
            optimizer.step()
             
        batcher.reset()
