import argparse
import multiprocessing
import os
import re

from qwikidata.linked_data_interface import get_entity_dict_from_api
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--language', choices=['it', 'en'], required=True)
parser.add_argument('--experiment_id', choices=['one', 'two'], required=True)
parser.add_argument('--corpora_folder', type=str, required=True)
args = parser.parse_args()

def process_entity(all_args):
    corpus = all_args[0]
    aliases = all_args[1]
    args = all_args[2]
    folder = args.corpora_folder
    ent_sentences = {e : list() for e in aliases.keys()}
    with tqdm() as counter:
        phrase_counter = 0
        with open(os.path.join(folder, '{}_{}.corpus'.format(corpus, args.language))) as i:
            for l in i.readlines():
                phrase_counter += 1
                #if phrase_counter > 500:
                #
                #    continue
                #else:
                for e, als in aliases.items():
                    marker = False
                    for al in als:
                        finder = re.findall('(?<!\w){}(?!\w)'.format(al), l)
                        if len(finder) >= 1:
                            marker = True
                    if marker == True:
                        new_sent = '{}'.format(l)
                        for al in als:
                            finder = re.findall('(?<!\w){}(?!\w)'.format(al), new_sent)
                            for match in finder:
                                new_sent = re.sub('(?<![a-zA-Z#]){}(?![a-zA-Z#])'.format(match), '[SEP]#{}#[SEP]'.format(match), new_sent)
                                #print(new_sent)
                            #new_sent = re.sub('\W{}\W'.format(al), ' [SEP]{}[SEP] '.format(al), new_sent)
                        new_sent = re.sub(r'#', r' ', new_sent)
                        new_sent = re.sub('\s\s', r' ', new_sent)
                        ent_sentences[e].append('{}\t{}'.format(corpus, new_sent.strip()))
                        counter.update(1)
    return ent_sentences

ent_sentences = dict()
aliases = dict()
with open('wikidata_ids_{}.txt'.format(args.experiment_id)) as i:
    lines = [tuple(l.strip().split('\t')) for l in i.readlines()]
lines = [l for l in lines if len(l)==2]
for e, wikidata_id in tqdm(lines):
    ent_sentences[e] = list()
    aliases[e] = list()
    ent_dict = get_entity_dict_from_api(wikidata_id)
    aliases[e].append(ent_dict['labels'][args.language]['value'])
    if args.language in ent_dict['aliases'].keys():
        for al in ent_dict['aliases'][args.language]:
            aliases[e].append(al['value'])
    '''
    ### Fix for body of water
    if 'acqua' in e:
        if args.language == 'it':
            words = ['mare', 'Mare', 'fiume', 'oceano', 'Oceano', 'lago']
        elif args.language == 'en':
            words = ['sea', 'Sea', 'river', 'ocean', 'Ocean', 'lake']
        aliases[e].extend(words)
    '''

correct_aliases = {k : list() for k in aliases.keys()}
for e, als in aliases.items():
    for l in als:
        l = re.sub('\[\d+\]', '', l)
        l = re.sub('[=_*#]', ' ', l)
        l = re.sub('\s+', ' ', l)
        l = re.sub(r'([<>"\',\.;?!:\(\)\[\]])', r' \1 ', l)
        l = re.sub('\s+', ' ', l)
        correct_aliases[e].append(l)

if args.experiment_id == 'one':
    correct_aliases['città']  = ['citt\w']

### Ordering aliases by length, so as to avoid problems
correct_aliases = {k : [v[1] for v in sorted([(len(val), val) for val in als], key=lambda item : item[0], reverse=True)] for k, als in correct_aliases.items()}

corpora = ['wiki', 'opensubtitles', 'itwac', 'gutenberg']

with multiprocessing.Pool(processes=len(corpora)) as pool:
   results = pool.map(process_entity, [(corpus, correct_aliases, args) for corpus in corpora])
   pool.terminate()
   pool.join()

out_folder = os.path.join('entity_sentences_{}_from_all_corpora'.format(args.experiment_id), args.language)
os.makedirs(out_folder, exist_ok=True)
final_sents = {k : list() for k in aliases.keys()}
for ent_dict in results:
    for k, v in ent_dict.items():
        final_sents[k].extend(v)

for stimulus, ent_sentences in final_sents.items():
    with open(os.path.join(out_folder, '{}.sentences'.format(stimulus)), 'w') as o:
        for sent in ent_sentences:
            o.write('{}\n'.format(sent.strip()))
