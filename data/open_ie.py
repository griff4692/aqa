from collections import defaultdict
import json
import os
import pickle
import re
import sys
from time import time

from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules, sanitize
import_submodules('imojie')
import matplotlib.pyplot as plt
from nltk import sent_tokenize
import torch

from utils import dict_to_lists, duration


def process_imojie_output(token_ids):
    temp = ' '.join(token_ids)
    temp = temp.replace(" ##","")
    temp = temp.replace("[unused1]","( ")
    temp = temp.replace("[unused2]"," ; ")
    temp = temp.replace('[unused3]',"")
    temp = temp.replace('[unused4]',' ; ')
    temp = temp.replace('[unused5]','')
    temp = temp.replace('[unused6]',' )')
    temp = temp.strip()
    temp = temp.split('[SEP]')
    ans = []
    for x in temp:
        if x != '':
            ans.append(x)
    return ans


def extract_tups_from_str(str):
    return list(map(lambda x: x.lower().strip(), str.strip('() ').split('  ;  ')))


def extract_ie(example):
    idx, example = example

    context = example['resolved']
    sents = sent_tokenize(context)
    ie_tups = []

    n = len(sents)
    max_batch_size = 500
    outputs = []
    for start_idx in range(0, n, max_batch_size):
        end_idx = min(start_idx + max_batch_size, n)
        batch_sents = sents[start_idx:end_idx]
        batch_inputs = list(map(predictor._dataset_reader.text_to_instance, batch_sents))
        batch_outputs = predictor._model.forward_on_instances(batch_inputs)
        batch_outputs = sanitize(batch_outputs)
        outputs += batch_outputs

    for output in outputs:
        output = process_imojie_output(output['predicted_tokens'][0])
        ie_tup = list(map(extract_tups_from_str, output))
        ie_tup_valid = list(filter(lambda x: len(x) == 3, ie_tup))
        ie_tups += ie_tup_valid
    example['ie_tups'] = ie_tups
    if (idx + 1) % update_incr == 0:
        print('Processed {} examples'.format(idx + 1))
    return example


def process_dataset(contexts, out_fn):
    keys, context_objects = dict_to_lists(contexts)
    ie_output = list(map(extract_ie, enumerate(context_objects)))
    output_dict = {}
    for k, v in zip(keys, ie_output):
        output_dict[k] = v
    print('Saving {} contexts with open IE tuples to {}...'.format(len(output_dict), out_fn))
    with open(out_fn, 'w') as fd:
        json.dump(output_dict, fd)


if __name__ == '__main__':
    debug_mode = len(sys.argv) > 1 and sys.argv[1] == 'debug'
    device = 1 if torch.cuda.is_available() else -1
    update_incr = 1 if debug_mode else 1000

    print('Loading Dataset')
    context_fn = 'trivia_qa/contexts_debug.json'
    with open(context_fn, 'r') as fd:
        contexts = json.load(fd)

    archive = load_archive(
        '../pretrained_models/imojie',
        weights_file='../pretrained_models/imojie/model_state_epoch_7.th',
        cuda_device=device)

    predictor = Predictor.from_archive(archive, 'noie_seq2seq')

    start_time = time()
    if debug_mode:
        in_fn = os.path.join('trivia_qa', 'contexts_debug.json')
        out_fn = os.path.join('trivia_qa', 'contexts_ie_debug.json')
        with open(in_fn, 'r') as fd:
            contexts = json.load(fd)
        process_dataset(contexts, out_fn)
    else:
        dtypes = ['test', 'train', 'validation']
        for dtype in dtypes:
            print('Extracting open IE tuples for {} set'.format(dtype))
            in_fn = os.path.join('trivia_qa', 'contexts_{}.json'.format(dtype))
            out_fn = os.path.join('trivia_qa', 'contexts_ie_{}.json'.format(dtype))
            with open(in_fn, 'r') as fd:
                contexts = json.load(fd)
            process_dataset(contets, out_fn)

    duration(start_time)
