import re

from allennlp.predictors.predictor import Predictor
import networkx as nx
import neuralcoref
import matplotlib.pyplot as plt

import spacy
from spacy.lang.en import English

ARG_REGEX = r'\[ARG(\d{1,3}: [^\[\]]+)\]'


def extract_args(description):
    arg_strs = re.findall(ARG_REGEX, description)
    results = []
    for arg_str in arg_strs:
        d, arg = arg_str.split(':')
        d = int(d)
        arg = arg.strip()
        results.append(arg)
    return results


if __name__== '__main__':
    
    #LINK THE INPUT PIPE FROM HERE (inpt):
        
    spacy_nlp = spacy.load('en_core_web_lg')
    coref = neuralcoref.NeuralCoref(spacy_nlp.vocab)
    spacy_nlp.add_pipe(coref, name='neuralcoref')    
    
    inpt = 'Mert is a PhD student.  He is from Turkey.'
    
    resolved_inpt = spacy_nlp(inpt)._.coref_resolved
        
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated
    split_resolved_inpt = nlp(resolved_inpt)

    sentences = [sent.string.strip() for sent in split_resolved_inpt.sents]
    
    predictor = Predictor.from_path(
        'https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz')

    g = nx.DiGraph()
    
    for sent in sentences:
    
        outputs = predictor.predict(
            sentence=sent
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
