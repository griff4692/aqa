import json
import os
import sys
from time import time

# from allennlp.predictors.predictor import Predictor
# from allennlp.models.archival import load_archive
# from allennlp.common.util import import_submodules, sanitize
# import_submodules('imojie')
import argparse
from nltk import sent_tokenize
import spacy
from textpipeliner import PipelineEngine, Context
from textpipeliner.pipes import *
import torch

from dataset_base import dataset_factory
from utils import dict_to_lists, duration


def process_imojie_output(token_ids):
    temp = ' '.join(token_ids)
    temp = temp.replace(" ##","")
    temp = temp.replace("[unused1]","( ")
    temp = temp.replace("[unused2]"," ; ")
    temp = temp.replace('[unused3]',"")
    temp = temp.replace('[unused4]',' ; ')
    temp = temp.replace('[unused5]','')
    temp = temp.replace('[unused6]',' )')
    temp = temp.strip()
    temp = temp.split('[SEP]')
    ans = []
    for x in temp:
        if x != '':
            ans.append(x)
    return ans


def extract_tups_from_str(str):
    return list(map(lambda x: x.lower().strip(), str.strip('() ').split('  ;  ')))


# def extract_dep_tups(spacy_doc):
#     engine = PipelineEngine(pipes_structure, Context(spacy_doc), [0, 1, 2])
#     process = engine.process()
#     print(spacy_doc, process)
#     return process

def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def extract_dep_tups(sent):
    ent_pairs = []
    spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
    spans = filter_spans(spans)
    with sent.retokenize() as retokenizer:
        [retokenizer.merge(span) for span in spans]
    dep = [token.dep_ for token in sent]
    if (dep.count('obj') + dep.count('dobj')) == 1 \
            and (dep.count('subj') + dep.count('nsubj')) == 1:
        for token in sent:
            if token.dep_ in ('obj', 'dobj'):  # identify object nodes
                subject = [w for w in token.head.lefts if w.dep_
                           in ('subj', 'nsubj')]  # identify subject nodes
                if subject:
                    subject = subject[0]
                    # identify relationship by root dependency
                    relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                    if relation:
                        relation = relation[0]
                        # add adposition or particle to relationship
                        if relation.nbor(1).pos_ in ('ADP', 'PART'):
                            relation = ' '.join((str(relation),
                                                 str(relation.nbor(1))))
                    else:
                        relation = 'unknown'
                    subject, subject_type = refine_ent(subject, sent)
                    token, object_type = refine_ent(token, sent)
                    ent_pairs.append([str(subject), str(relation), str(token),
                                      str(subject_type), str(object_type)])
    filtered_ent_pairs = [sublist for sublist in ent_pairs if not any(str(x) == '' for x in sublist)]
    print(ent_pairs)
    return filtered_ent_pairs


def refine_ent(ent, sent):
    unwanted_tokens = (
        'PRON',  # pronouns
        'PART',  # particle
        'DET',  # determiner
        'SCONJ',  # subordinating conjunction
        'PUNCT',  # punctuation
        'SYM',  # symbol
        'X',  # other
    )
    ent_type = ent.ent_type_  # get entity type
    if ent_type == '':
        ent_type = 'NOUN_CHUNK'
        ent = ' '.join(str(t.text) for t in
                       spacy_nlp(str(ent)) if t.pos_
                       not in unwanted_tokens and t.is_stop == False)
    elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
        t = ''
        for i in range(len(sent) - ent.i):
            if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                t += ' ' + str(ent.nbor(i))
            else:
                ent = t.strip()
                break
    return ent, ent_type


def extract_spacy_ie(example):
    idx, example = example
    context = example['resolved']
    sents = sent_tokenize(context)

    docs = [spacy_nlp(s) for s in sents]
    dep_tups = list(map(extract_dep_tups, docs))

    example['ie_tups'] = dep_tups
    if (idx + 1) % update_incr == 0:
        print('Processed {} examples'.format(idx + 1))
    return example


def extract_emoji_ie(example):
    idx, example = example
    context = example['resolved']
    sents = sent_tokenize(context)
    ie_tups = []

    n = len(sents)
    max_batch_size = 400
    outputs = []
    for start_idx in range(0, n, max_batch_size):
        end_idx = min(start_idx + max_batch_size, n)
        batch_sents = sents[start_idx:end_idx]
        batch_inputs = list(map(predictor._dataset_reader.text_to_instance, batch_sents))
        batch_outputs = predictor._model.forward_on_instances(batch_inputs)
        batch_outputs = sanitize(batch_outputs)
        outputs += batch_outputs

    for output in outputs:
        output = process_imojie_output(output['predicted_tokens'][0])
        ie_tup = list(map(extract_tups_from_str, output))
        ie_tup_valid = list(filter(lambda x: len(x) == 3, ie_tup))
        ie_tups += ie_tup_valid
    example['ie_tups'] = ie_tups
    if (idx + 1) % update_incr == 0:
        print('Processed {} examples'.format(idx + 1))
    return example


def extract_relations(args, dataset, dtype):
    extractor = extract_spacy_ie if args.method == 'dep' else extract_emoji_ie
    in_fn = os.path.join('..', 'data', dataset.name, 'contexts_{}.json'.format(dtype))
    out_fn = os.path.join('..', 'data', dataset.name, 'contexts_ie_{}.json'.format(dtype))
    with open(in_fn, 'r') as fd:
        context_resolved = json.load(fd)

    keys, context_objects = dict_to_lists(context_resolved)
    ie_output = list(map(extractor, enumerate(context_objects)))
    output_dict = {}
    for k, v in zip(keys, ie_output):
        output_dict[k] = v
    print('Saving {} contexts with open IE tuples to {}...'.format(len(output_dict), out_fn))
    with open(out_fn, 'w') as fd:
        json.dump(output_dict, fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Coreference Resolution Preprocessing Script.')
    parser.add_argument('--method', default='dep', help='dep(endency parse) or open_ie')
    parser.add_argument('--dataset', default='hotpot_qa', help='trivia_qa or hotpot_qa')
    parser.add_argument(
        '-debug', default=False, action='store_true', help='If true, run on tiny portion of train dataset')
    args = parser.parse_args()

    dataset = dataset_factory(args.dataset)

    device = 0 if torch.cuda.is_available() else -1
    update_incr = 10 if args.debug else 10000

    if args.method == 'dep':
        # pipes_structure = [
        #     SequencePipe([
        #         FindTokensPipe('VERB/nsubj/*'),
        #         NamedEntityFilterPipe(),
        #         NamedEntityExtractorPipe()]),
        #     FindTokensPipe('VERB'),
        #
        #     AnyPipe([
        #         SequencePipe([FindTokensPipe('VBD/dobj/NNP'),
        #                       AggregatePipe([NamedEntityFilterPipe('GPE'),
        #                                      NamedEntityFilterPipe('PERSON')]),
        #                       NamedEntityExtractorPipe()]),
        #
        #         SequencePipe([FindTokensPipe('VBD/**/*/pobj/NNP'),
        #                       AggregatePipe([NamedEntityFilterPipe('LOC'),
        #                                      NamedEntityFilterPipe('PERSON')]),
        #                       NamedEntityExtractorPipe()])])]
        spacy_nlp = spacy.load('en_core_web_lg')
    else:
        # archive = load_archive(
        #     '../pretrained_models/imojie',
        #     weights_file='../pretrained_models/imojie/model_state_epoch_7.th',
        #     cuda_device=device)
        #
        # predictor = Predictor.from_archive(archive, 'noie_seq2seq')
        pass

    dtypes = ['mini'] if args.debug else ['train', 'test', 'validation']
    for dtype in dtypes:
        start_time = time()
        extract_relations(args, dataset, dtype)
        duration(start_time)
