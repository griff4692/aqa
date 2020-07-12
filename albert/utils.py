# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:45:47 2020

@author: Mert Ketenci
"""

def flat_integer_list(nested_integer_list):
    x = []
    for i in nested_integer_list:
        x = x + i
    return x


def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
        
    