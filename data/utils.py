from time import time


def duration(start_time):
    end_time = time()
    minutes = (end_time - start_time) / 60.0
    round_factor = 0
    if minutes < 1:
        round_factor = 2
    print('Took {} minutes'.format(minutes, round(round_factor)))


def _extract_contexts(example):
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


def extract_contexts(dataset):
    d = {}
    for i, example in enumerate(dataset):
        contexts = _extract_contexts(example)
        for k, v in contexts:
            if k in d:
                assert v == d[k]
            else:
                d[k] = v
        if (i + 1) % 10000 == 0 or (i + 1) == len(dataset):
            print('Loaded contexts for {} examples.'.format(i + 1))
    print('Unique documents={}'.format(len(d)))
    return d


def dict_to_lists(obj):
    keys, values = [], []
    for k, v in obj.items():
        keys.append(k)
        values.append(v)
    return keys, values
