import json
import os
import re
from time import time

from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
from bs4 import BeautifulSoup
import argparse
from tqdm import tqdm

from dataset_base import dataset_factory
from utils import duration


def clean(str):
    str = BeautifulSoup(str, 'html.parser').get_text(strip=True)
    str = str.replace('–', '-')
    str = re.sub(r'\s+', ' ', str)
    return str.strip()


def generate_coref_input(dataset, dtype, data_dir):
    """
    :param dataset_name: string name of dataset (path where it's saved in ../data)
    :param dtype: one of 'mini', 'train', 'validation', 'test'
    :return: None
    """
    print('Loading {} set...'.format(dtype))
    keys, texts = dataset.get_context_kv_pairs(dtype)
    n = len(keys)
    print('Processing {} contexts for {} set...'.format(len(keys), dtype))
    outputs = []
    for i in tqdm(range(n)):
        text_cleaned = clean(texts[i])
        outputs.append(text_cleaned)

    out_fn = os.path.join(data_dir, 'coref_output_{}.json')
    with open(out_fn, 'w') as fd:
        json.dump(outputs, fd)

    keys_out_fn = os.path.join(data_dir, 'coref_keys_{}.json')
    with open(keys_out_fn, 'w') as fd:
        json.dump(keys, fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate json file for  consumption by SpanBERT.')
    parser.add_argument('--dataset', default='squad', help='squad, trivia_qa, or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")

    data_dir = os.path.join('..', 'data', args.dataset, 'coref_data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    dataset = dataset_factory(args.dataset)
    if dataset.name == 'squad':
        dtypes = ['mini'] if args.debug else ['train', 'validation']
    else:
        dtypes = ['mini'] if args.debug else ['train', 'test', 'validation']

    for dtype in dtypes:
        start_time = time()
        generate_coref_input(dataset, dtype, data_dir)
        duration(start_time)
