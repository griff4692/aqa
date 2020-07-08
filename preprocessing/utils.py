import re
from time import time


def dict_to_lists(obj):
    keys, values = [], []
    for k, v in obj.items():
        keys.append(k)
        values.append(v)
    return keys, values


def duration(start_time):
    end_time = time()
    minutes = (end_time - start_time) / 60.0
    round_factor = 0
    if minutes < 1:
        round_factor = 2
    print('Took {} minutes'.format(minutes, round(round_factor)))


def remove_extra_space(str):
    """
    :param str: string
    :return: replace multiple spaces/newlines with a single space
    """
    return re.sub(r'\s+', ' ', str)