import json
import os
import re
from time import time

from bs4 import BeautifulSoup
import argparse
import spacy
from spacy.lang.en import English


from utils import duration

eos_tokens = '.!?'


def clean(str):
    str = BeautifulSoup(str, 'html.parser').get_text(strip=True)
    str = str.replace('–', '-')
    str = re.sub(r'\s+', ' ', str)
    return str.strip()


def generate_ie_input(dataset_name, dtype, data_dir):
    """
    :param dataset_name: string name of dataset (path where it's saved in ../data)
    :param dtype: one of 'mini', 'train', 'validation', 'test'
    :return: None

    """
    print('Loading {} set...'.format(dtype))
    context_fn = os.path.join('..', 'data', dataset_name, 'contexts_{}.json'.format(dtype))

    with open(context_fn, 'r') as fd:
        contexts = json.load(fd)
    print('Processing {} contexts for {} set...'.format(len(contexts), dtype))
    keys = []
    sents = []
    n = len(contexts)
    keys = contexts.keys()
    for i, (k, context_obj) in enumerate(contexts.items()):
        doc = spacy_nlp(clean(context_obj['resolved']))
        for j, sent in enumerate(doc.sents):
            sent = sent.string.strip()
            if len(sent) > 0 and len(re.sub(r'\W+', '', sent)) > 0:
                if j > 0 and len(sents) > 0 and not sents[-1][-1] in eos_tokens:
                    sents[-1] += ' {}'.format(sent)
                else:
                    keys.append(k)
                    sents.append(sent)
        if (i + 1) % 10000 == 0 or (i + 1) == n:
            print('Processed {} out of {} contexts.'.format(i + 1, n))

    sent_out_fn = os.path.join(data_dir, 'sentences_{}.txt'.format(dtype))
    with open(sent_out_fn, 'w') as fd:
        fd.write('\n'.join(sents))
    keys_out_fn = os.path.join(data_dir, 'keys_{}.txt'.format(dtype))
    with open(keys_out_fn, 'w') as fd:
        fd.write('\n'.join(keys))
    print('Saved sentences and corresponding context keys to {} and {}'.format(sent_out_fn, keys_out_fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate raw sentences file for consumption by open IE 6.')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    update_incr = 10 if args.debug else 10000
    print('Loading Spacy...')
    spacy_nlp = English()  # just the language with no model
    sentencizer = spacy_nlp.create_pipe('sentencizer')
    spacy_nlp.add_pipe(sentencizer)
    print('Done Loading Spacy...')

    data_dir = os.path.join('..', 'data', args.dataset, 'open_ie_data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    dtypes = ['mini'] if args.debug else ['train', 'test', 'validation']
    for dtype in dtypes:
        start_time = time()
        generate_ie_input(args.dataset, dtype, data_dir)
        duration(start_time)
