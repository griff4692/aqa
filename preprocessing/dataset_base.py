from collections import defaultdict
import json
import os
import string
import unicodedata

from nlp import load_dataset
from torch.utils.data import Subset
from tqdm import tqdm

home_dir = os.path.expanduser('~/aqa')
printable = set(string.printable)


def dict_to_lists(obj):
    keys, values = [], []
    for k, v in obj.items():
        keys.append(k)
        values.append(v)
    return keys, values


def dataset_factory(name):
    if name == 'hotpot_qa':
        return HotpotQA()
    elif name == 'trivia_qa':
        return TriviaQA()
    elif name == 'squad':
        return Squad()
    raise Exception('Didn\'t recognize dataset={}'.format(name))


class DatasetBase:
    """
    Abstract base class for Dataset Wrappers
    """
    def __init__(self, name):
        self.name = name
        self.data_dir = os.path.join('../data', name)
        if not os.path.exists(self.data_dir):
            print('Creating directory for {} in {}'.format(self.name, self.data_dir))
            os.mkdir(self.data_dir)

    def qid_key(self):
        return 'id'

    def get_mini(self):
        debug_data_fn = os.path.join(self.data_dir, 'train_mini.json')
        if os.path.exists(debug_data_fn):
            with open(debug_data_fn, 'r') as fd:
                dataset = json.load(fd)
        else:
            dataset = self.get_train()
            dataset = [dict(e) for e in Subset(dataset, list(range(10)))]
            with open(debug_data_fn, 'w') as fd:
                json.dump(dataset, fd)
        print('Returning mini dataset with {} examples'.format(len(dataset)))
        return dataset

    def get_train(self):
        raise Exception('Should be overwritten')

    def get_validation(self):
        raise Exception('Should be overwritten')

    def get_test(self):
        raise Exception('Should be overwritten')

    def __getitem__(self, item):
        if item == 'train':
            return self.get_train()
        elif item == 'mini':
            return self.get_mini()
        elif item == 'validation':
            return self.get_validation()
        else:
            return self.get_test()

    def get_train(self):
        return self.cached_dataset['train']

    def get_validation(self):
        return self.cached_dataset['validation']

    def get_test(self):
        return self.cached_dataset['test']

    def get_context_kv_pairs(self, type, skip_keys=[]):
        return dict_to_lists(self.extract_contexts(type, skip_keys=skip_keys))


class HotpotQA(DatasetBase):
    def __init__(self):
        super().__init__('hotpot_qa')

    def qid_key(self):
        return '_id'

    def question_key(self):
        return 'question'

    def answer_keys(self):
        return ['answer']

    def get_linked_contexts(self, dtype):
        linked = defaultdict(list)
        examples = self[dtype]
        for example in examples:
            id = example['_id']
            context_ids = list(map(lambda x: x[0], example['context']))
            linked[id] = context_ids
        return linked

    def extract_contexts(self, dtype, skip_keys=[]):
        d = {}
        examples = self[dtype]
        skip_keys = set(skip_keys)
        skipped = set()
        n = len(examples)
        for i in tqdm(range(n)):
            for context in examples[i]['context']:
                k, v = context[0], unicodedata.normalize('NFKD', ' '.join(context[1]))
                if k in skip_keys:
                    skipped.add(k)
                    continue
                if k in d:
                    assert v == d[k]
                else:
                    d[k] = v
        assert len(skip_keys) == len(skipped)
        return d

    def remove_q_types(self, examples, qtypes=['comparison']):
        return list(filter(lambda x: 'type' not in x or not x['type'] in qtypes, examples))

    def get_train(self):
        with open(os.path.join(self.data_dir, 'hotpot_train_v1.1.json'), 'r') as fd:
            return self.remove_q_types(json.load(fd))

    def get_validation(self):
        with open(os.path.join(self.data_dir, 'hotpot_dev_fullwiki_v1.json'), 'r') as fd:
            return self.remove_q_types(json.load(fd))

    def get_test(self):
        with open(os.path.join(self.data_dir, 'hotpot_test_fullwiki_v1.json'), 'r') as fd:
            return self.remove_q_types(json.load(fd))


class Squad(DatasetBase):
    def __init__(self):
        super().__init__('squad')
        self.cached_dataset = load_dataset('squad')

    def question_key(self):
        return 'question'

    def answer_keys(self):
        return ['answers', 'text', 0]

    def get_linked_contexts(self, dtype):
        linked = defaultdict(list)
        examples = self[dtype]
        for example in examples:
            id = example['id']
            v = example['context'].strip()
            k = '{}_{}_{}_{}'.format(example['title'], v[:10], v[-10:], str(len(v)))
            linked[id] = [k]
        return linked

    def extract_contexts(self, type, skip_keys=[]):
        d = {}
        s = set()
        skip_keys = set(skip_keys)
        examples = self[type]
        n = len(examples)
        for i in tqdm(range(n)):
            example = examples[i]
            v = example['context'].strip()
            k = '{}_{}_{}_{}'.format(example['title'], v[:10], v[-10:], str(len(v)))
            s.add(v)
            if k in skip_keys:
                continue
            v = ''.join(filter(lambda x: x in printable, v))
            if k in d:
                assert v == d[k]
            else:
                d[k] = v
        print('Unique documents={}'.format(len(d)))
        return d


class TriviaQA(DatasetBase):
    def __init__(self):
        super().__init__('trivia_qa')
        self.cached_dataset = load_dataset('trivia_qa', 'rc')

    def _extract_contexts(self, example):
        keys = []
        texts = []
        for title, text in zip(example['entity_pages']['title'], example['entity_pages']['wiki_context']):
            k = '{}_{}'.format('wiki', title)
            keys.append(k)
            texts.append(text)

        for fn, text in zip(example['search_results']['url'], example['search_results']['search_context']):
            k = '{}_{}'.format('search', fn)
            keys.append(k)
            texts.append(text)

        return list(zip(keys, texts))

    def extract_contexts(self, type, skip_keys=[]):
        d = {}
        skip_keys = set(skip_keys)
        examples = self[type]
        n = len(examples)
        for i in tqdm(range(n)):
            example = examples[i]
            contexts = self._extract_contexts(example)
            for k, v in contexts:
                if k in skip_keys:
                    continue

                v = ''.join(filter(lambda x: x in printable, v))
                if k in d:
                    assert v == d[k]
                else:
                    d[k] = v
        print('Unique documents={}'.format(len(d)))
        return d
