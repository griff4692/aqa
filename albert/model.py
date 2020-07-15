# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:28:51 2020

@author: Mert Ketenci
"""
import torch
from torch import nn
from torch.nn import Flatten
from torch.nn.functional import cross_entropy
from transformers import AlbertModel
from transformers import AlbertConfig
import os
from pathlib import Path
from utils import overlap

class AlbertQAModel(nn.Module):
    def __init__(self, max_pos = 512):
        
        super(AlbertQAModel, self).__init__()
        
        if max_pos > 512:
            self.encoder = AlbertModel.from_pretrained(path + f'/albert-longform')
        else:
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
            
        hidden_size = self.encoder.encoder.embedding_hidden_mapping_in.out_features
        
        self.start_logits = nn.Linear(max_pos * hidden_size, max_pos, bias = True)
        self.end_logits = nn.Linear(max_pos * hidden_size, max_pos, bias = True)
        
    def forward(self, x_train = None, y_train = None, token_type = None, 
                x_test = None, y_context = None, y_answer = None, is_train = True):
        
        if is_train:
            embeddings = self.encoder(x_train, token_type_ids = token_type)[0]
            embeddings = Flatten()(embeddings)
            
            beg = self.start_logits(embeddings)
            end = self.end_logits(embeddings)
            
            loss_beg = cross_entropy(beg, y_train[:,0])
            loss_end = cross_entropy(end, y_train[:,1])
            return loss_beg + loss_end
        else:
            embeddings = self.encoder(x_test, token_type_ids = token_type)[0]
            embeddings = Flatten()(embeddings)
            
            beg = torch.argmax(self.start_logits(embeddings), dim=1)
            end = torch.argmax(self.end_logits(embeddings), dim=1)
            
            exact_match = overlap(beg, end, y_context, y_answer)
                
                
                
            return exact_match
