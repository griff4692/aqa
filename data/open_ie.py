import re

from allennlp.predictors.predictor import Predictor
import networkx as nx
import matplotlib.pyplot as plt


ARG_REGEX = r'\[ARG(\d{1,3}: [^\[\]]+)\]'


def extract_args(description):
    arg_strs = re.findall(ARG_REGEX, description)
    results = [None] * len(arg_strs)
    for arg_str in arg_strs:
        d, arg = arg_str.split(':')
        d = int(d)
        arg = arg.strip()
        results[d] = arg
    return results


if __name__== '__main__':
    predictor = Predictor.from_path(
        'https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz')

    g = nx.DiGraph()

    outputs = predictor.predict(
        sentence='John decided to run for office next month.'
    )['verbs']
    for output in outputs:
        verb = output['verb']
        args = extract_args(output['description'])
        subject = args[0]
        objects = args[1:]

        for obj in objects:
            print(subject, obj, verb)
            g.add_edge(subject, obj, weight=1, predicate=verb)
    print(g.nodes())
    pos = nx.spring_layout(g)
    nx.draw(g, pos=pos)
    nx.draw_networkx_labels(g, pos=pos)
    nx.draw_networkx_edge_labels(g, pos=pos)

    plt.savefig('tmp.png')
    plt.show()
