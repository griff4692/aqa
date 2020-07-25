from collections import defaultdict
import json
import os
import re
from time import time

import argparse
import nltk

from utils import duration

MATCH_REGEX = r'\d\.\d+: \((.*)\)$'
MINI_MATCH_REGEX = r'[.\d]+. \((.*)\)'


def process_source(str):
    source_sentence = str.replace('’', '\'')
    source_sentence = source_sentence.replace('”', '\'\'')
    source_sentence = source_sentence.replace('“', '\'\'')
    source_sentence = nltk.tokenize.word_tokenize(source_sentence.strip('\n'))
    target_sentence = ' '.join(source_sentence[:95])
    return target_sentence


def process_output(dataset, dtype):
    regex = MINI_MATCH_REGEX if dtype == 'mini' else MATCH_REGEX
    oie_sentences_fn = os.path.join('..', 'data', dataset, 'oie_data', 'sentences_{}.txt'.format(dtype))

    source = []
    with open(oie_sentences_fn, 'r') as fd:
        for line in fd:
            line = line.strip()
            if len(line) > 0:
                source.append(process_source(line))

    source_set = set(source)
    oie_results_fn = os.path.join('..', 'data', dataset, 'oie_data', 'predictions_{}.txt'.format(dtype))
    out_fn = os.path.join('..', 'data', dataset, 'oie_data', 'predictions_{}.json'.format(dtype))
    keys_fn = os.path.join('..', 'data', dataset, 'oie_data', 'keys_{}.json'.format(dtype))

    with open(keys_fn, 'r') as fd:
        keys = json.load(fd)
    num_keys = len(keys)
    results = []
    curr_result = []

    with open(oie_results_fn, 'r') as fd:
        for i, line in enumerate(fd):
            line = line.strip()
            if line in source_set:
                if i > 0:
                    results.append(curr_result)
                    curr_result = []
            elif len(line) > 0:
                ie_match = re.match(regex, line, flags=re.M)
                ie_output = ie_match.group(1).split(';')
                ie_output = list(map(lambda x: x.strip(), ie_output))
                ie_output_stripped = list(filter(lambda x: len(x) > 0, ie_output))
                if len(ie_output_stripped) == 3:
                    curr_result.append(ie_output_stripped)
    results.append(curr_result)
    num_result = len(results)
    print('Processed outputs for {} sentences'.format(num_result))
    if not num_keys == num_result:
        raise Exception('Processed ie_tups for {} sentences.  Supposed to be {} contexts'.format(num_result, num_keys))
    output = defaultdict(list)
    for k, o in zip(keys, results):
        output[k] += o
    print('Saving {} distinct contexts to {}'.format(len(output), out_fn))
    with open(out_fn, 'w') as fd:
        json.dump(output, fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate raw sentences file for consumption by open IE 6.')
    parser.add_argument('--dataset', default='squad', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    if args.dataset == 'squad':
        dtypes = ['mini'] if args.debug else ['train', 'validation']
    else:
        dtypes = ['mini'] if args.debug else ['train', 'test', 'validation']

    results = []
    for dtype in dtypes:
        start_time = time()
        process_output(args.dataset, dtype)
        duration(start_time)
