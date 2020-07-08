import json
import os
import re
from time import time
import argparse
import pandas as pd
from utils import duration

def process_output(dataset, dtype):
    oie_results_fn = os.path.join('..', 'data', dataset, 'open_ie_data', 'predictions_{}.txt'.format(dtype))
    out_fn = os.path.join('..', 'data', dataset, 'open_ie_data', 'predictions_{}.json'.format(dtype))
    results = []
    curr_result = []
    prev_line = 'random'
    with open(oie_results_fn, 'r') as fd:
        for line in fd:
            line = line.strip()
            ie_match = re.match(r'\d\.\d\d: \((.*)\)$', line, flags=re.M)
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
    print('Saved results for {} sentences'.format(num_result))
    with open(out_fn, 'w') as fd:
        json.dump(results, fd)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate raw sentences file for consumption by open IE 6.')
    parser.add_argument('--dataset', default='hotpot_qa',
                        help='trivia_qa or hotpot_qa')
    parser.add_argument('-debug', default=False, action='store_true', 
                        help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()
    dtypes = ['mini'] if args.debug else ['train', 'test', 'validation']
    results = []
    for dtype in dtypes:
        start_time = time()
        process_output(args.dataset, dtype)
        duration(start_time)