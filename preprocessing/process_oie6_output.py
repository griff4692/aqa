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
MIN_CONFIDENCE = 0.01


def process_output(dataset, dtype):
    regex = MINI_MATCH_REGEX if dtype == 'mini' else MATCH_REGEX
    oie_results_fn = os.path.join('..', 'data', dataset, 'oie_data', 'predictions_{}.txt'.format(dtype))
    out_fn = os.path.join('..', 'data', dataset, 'oie_data', 'predictions_{}.json'.format(dtype))
    keys_fn = os.path.join('..', 'data', dataset, 'oie_data', 'keys_{}.json'.format(dtype))

    with open(keys_fn, 'r') as fd:
        keys = json.load(fd)
    num_keys = len(keys)
    results = []
    curr_result = []

    with open(oie_results_fn, 'r') as fd:
        prev_line = 'random'
        for i, line in enumerate(fd):
            line = line.strip()
            if len(line) == 0 and not len(prev_line) == 0:
                results.append(curr_result)
                curr_result = []
            else:
                ie_match = re.match(regex, line, flags=re.M)
                if ie_match is not None:
                    ie_output = ie_match.group(1).split(';')
                    ie_output = list(map(lambda x: x.strip(), ie_output))
                    ie_output_stripped = list(filter(lambda x: len(x) > 0, ie_output))
                    conf = float(line[:4])
                    should_include = len(curr_result) < 3 or float(conf) > MIN_CONFIDENCE
                    if len(ie_output_stripped) == 3 and should_include:
                        curr_result.append(ie_output_stripped)
            prev_line = line
    if len(prev_line) > 0:
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
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    if args.dataset == 'squad':
        dtypes = ['mini'] if args.debug else ['validation', 'train']
    else:
        dtypes = ['mini'] if args.debug else ['test', 'validation', 'train']

    results = []
    for dtype in dtypes:
        start_time = time()
        print('Processing {} {} set...'.format(args.dataset, dtype))
        process_output(args.dataset, dtype)
        duration(start_time)
