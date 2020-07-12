# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:28:51 2020

@author: Mert Ketenci
"""

from transformers import AlbertModel

class AlbertQAModel:
    def __init__(self):
        pass
    
    def forward(self, x_train, x_test, x_token_type_ids, y_train, y_test, y_token_type_ids):
        
        encoder = AlbertModel.from_pretrained('albert-base-v2')
        embedding = encoder(x_train, token_type_ids = x_token_type_ids)[0]
        
        #Implement fine-tuning network starting from here