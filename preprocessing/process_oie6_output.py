from collections import defaultdict
import json
import os
import re
from time import time

import argparse

from utils import duration

MATCH_REGEX = r'\d\.\d+: \((.*)\)$'
MINI_MATCH_REGEX = r'[.\d]+. \((.*)\)'


def process_output(dataset, dtype):
    regex = MINI_MATCH_REGEX if dtype == 'mini' else MATCH_REGEX
    oie_results_fn = os.path.join('..', 'data', dataset, 'open_ie_data', 'predictions_{}.txt'.format(dtype))
    out_fn = os.path.join('..', 'data', dataset, 'open_ie_data', 'predictions_{}.json'.format(dtype))
    keys_fn = os.path.join('..', 'data', dataset, 'open_ie_data', 'keys_{}.txt'.format(dtype))
    keys = list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(), open(keys_fn, 'r').readlines())))
    num_keys = len(keys)
    results = []
    curr_result = []
    prev_line = 'random'
    with open(oie_results_fn, 'r') as fd:
        for line in fd:
            line = line.strip()
            ie_match = re.match(regex, line, flags=re.M)
            if len(line) == 0 and not len(prev_line) == 0:
                results.append(curr_result)
                curr_result = []
            elif ie_match is not None:
                ie_output = ie_match.group(1).split(';')
                ie_output = list(map(lambda x: x.strip(), ie_output))
                ie_output_stripped = list(filter(lambda x: len(x) > 0, ie_output))
                if len(ie_output_stripped) == 3:
                    curr_result.append(ie_output_stripped)
            prev_line = line

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

    dtypes = ['mini'] if args.debug else ['train', 'test', 'validation']
    results = []
    for dtype in dtypes:
        start_time = time()
        process_output(args.dataset, dtype)
        duration(start_time)
