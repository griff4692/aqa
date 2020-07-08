import json
import os

from nlp import load_dataset
from torch.utils.data import Subset

home_dir = os.path.expanduser('~/aqa')


def dict_to_lists(obj):
    keys, values = [], []
    for k, v in obj.items():
        keys.append(k)
        values.append(v)
    return keys, values


def dataset_factory(name):
    if name == 'hotpot_qa':
        return HotpotQA()
    else:
        return TriviaQA()


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


class HotpotQA(DatasetBase):
    def __init__(self):
        super().__init__('hotpot_qa')

    def get_context_kv_pairs(self, type):
        keys, texts = [], []
        for example in self[type]:
            for context in example['context']:
                keys.append(context[0])
                # Given as list of sentences.  need one passage for proper coref resolution
                texts.append(' '.join(context[1]))
        return keys, texts

    def remove_q_types(self, examples, qtypes=['comparison']):
        return list(filter(lambda x: not x['type'] in qtypes, examples))

    def get_train(self):
        with open(os.path.join(self.data_dir, 'hotpot_train_v1.1.json'), 'r') as fd:
            return self.remove_q_types(json.load(fd))

    def get_validation(self):
        with open(os.path.join(self.data_dir, 'hotpot_dev_fullwiki_v1.json'), 'r') as fd:
            return self.remove_q_types(json.load(fd))

    def get_test(self):
        with open(os.path.join(self.data_dir, 'hotpot_test_fullwiki_v1.json'), 'r') as fd:
            return self.remove_q_types(json.load(fd))


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

    def extract_contexts(self, type):
        d = {}
        for i, example in enumerate(self[type]):
            contexts = self._extract_contexts(example)
            for k, v in contexts:
                if k in d:
                    assert v == d[k]
                else:
                    d[k] = v
            if (i + 1) % 10000 == 0 or (i + 1) == len(self[type]):
                print('Loaded contexts for {} examples.'.format(i + 1))
        print('Unique documents={}'.format(len(d)))
        return d

    def get_context_kv_pairs(self, type):
        return dict_to_lists(self.extract_contexts(type))

    def get_train(self):
        return self.cached_dataset['train']

    def get_validation(self):
        return self.cached_dataset['validation']

    def get_test(self):
        return self.cached_dataset['test']
