# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:28:51 2020

@author: Mert Ketenci
"""

from transformers import AlbertModel
from transformers import AlbertConfig

AlbertConfig (max_position_embeddings = 10000)

model = AlbertModel
config.__dict__['max_position_embeddings'] = 2048



class AlbertQAModel:
    def __init__(self, len_max_seq):
        self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        
        self.start_layer = nn.Linear(len_max_seq, len_max_seq, bias = True)
        self.start_logits = nn.Linear(len_max_seq, len_max_seq, bias = True)
        
        self.end_layer = nn.Linear(len_max_seq, len_max_seq, bias = True)
        self.end_logits = nn.Linear(len_max_seq, len_max_seq, bias = True)
        
    def forward(self, x_train, x_test, token_type, y_train, y_test, y_token_type_ids):
        
        embeddings = self.encoder(x_train, token_type_ids = token_type)[0]
        embeddings = torch.flatten(embeddings)
        start = self.start_logits(self.start_logits(embeddings))
        end = self.end_logits(self.end_layer(embeddings))
        
        return start, end