# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:28:51 2020

@author: Mert Ketenci
"""
from torch import nn
from torch.nn.functional import cross_entropy
from transformers import AlbertModel
from transformers import AlbertConfig

class AlbertQAModel(nn.Module):
    def __init__(self, max_seq_len = 512):
        
        super(AlbertQAModel, self).__init__()
        
        self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        hidden_size = self.encoder.encoder.embedding_hidden_mapping_in.out_features
        
        # self.start_layer = nn.Linear(max_seq_len * hidden_size, max_seq_len, bias = True)
        self.start_logits = nn.Linear(max_seq_len * hidden_size, max_seq_len, bias = True)
        
        # self.end_layer = nn.Linear(max_seq_len * hidden_size, max_seq_len, bias = True)
        self.end_logits = nn.Linear(max_seq_len * hidden_size, max_seq_len, bias = True)
        
    def forward(self, x_train, y_train, token_type, x_test = None, y_test = None, is_train = True):
        if is_train:
            embeddings = self.encoder(x_train, token_type_ids = token_type)[0]
            embeddings = torch.flatten(embeddings)
            
            start = self.start_logits(embeddings)
            end = self.end_logits(embeddings)
            
            loss_start = cross_entropy(start, y_train[:,0])
            loss_end = cross_entropy(end, y_train[:,1])
        
            return loss_start, loss_end
        else:
            embeddings = self.encoder(x_test, token_type_ids = token_type)[0]
            embeddings = torch.flatten(embeddings)
            
            start = self.start_logits(embeddings)
            end = self.end_logits(embeddings)
            
            loss_start = cross_entropy(start, y_test[:,0])
            loss_end = cross_entropy(end, y_test[:,1])
            return loss_start, loss_end
