# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:14:59 2020

@author: Mert Ketenci

modified from : https://colab.research.google.com/github/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb#scrollTo=c7tcsSfZ1-b9

"""

import argparse
from pathlib import Path
import pickle
import sys,os

from transformers import AlbertForMaskedLM
from transformers import logging
from transformers.modeling_longformer import LongformerSelfAttention


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

            
def create_long_model(save_model_to, attention_window, max_pos):
    model = AlbertForMaskedLM.from_pretrained('albert-base-v2')
    config = model.config

    # extend position embeddings
    current_max_pos, embed_size = model.albert.embeddings.position_embeddings.weight.shape

    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.albert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.albert.embeddings.position_embeddings.weight
        k += step
    model.albert.embeddings.position_embeddings.weight.data = new_pos_embed
        
    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.albert.encoder.albert_layer_groups[0].albert_layers):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.query
        longformer_self_attn.key = layer.attention.key
        longformer_self_attn.value = layer.attention.value

        longformer_self_attn.query_global = layer.attention.query
        longformer_self_attn.key_global = layer.attention.key
        longformer_self_attn.value_global = layer.attention.value

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    return model


path = os.path.join(Path(os.path.dirname(__file__)))
os.chdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build the longtransformer albert')

    parser.add_argument('-max_pos', default = 4096, help = 'Longformer adjusted max albert sequence length')
    parser.add_argument('-attention_window', default = 512, help = 'Attention window chunks')

    args = parser.parse_args()
        
    model_path = path + f'/albert-longform'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    model = create_long_model(save_model_to = model_path, 
                                         attention_window = args.attention_window, 
                                         max_pos = args.max_pos)