import os
import re
import collections
import torch
import numpy
import argparse

from transformers import BertModel, BertTokenizer
from utils.extract_word_lists import Entities
from tqdm import tqdm

def vector_to_txt(word, vector, output_file):
    output_file.write('{}\t'.format(word))
    for dimension_index, dimension_value in enumerate(vector):
        if dimension_index != len(vector)-1:
            output_file.write('{}\t'.format(dimension_value))
        else: 
            output_file.write('{}\n'.format(dimension_value))

# Create multiple vectors for Bert clustering analysis

parser = argparse.ArgumentParser()
parser.add_argument('--entities', choices=['full_wiki', 'wakeman_henson', 'eeg_stanford', 'mitchell'], default='full_wiki', help='Indicates which entities should be extracted')
args = parser.parse_args()

'''
if args.entities == 'full_wiki':
    ents = [k for k, v in Entities('full_wiki').words[1]]
elif args.entities == 'wakeman_henson':
    #ents = [v for k, v in (Entities('wakeman_henson').words)[1].items()]
    ents = Entities(args.entities).words
elif args.entities == 'eeg_stanford':
    ents, _ = Entities(args.entities).words
    ents = [w for w in ents if len(w)>2]
elif args.entities == 'mitchell':
    ents = Entities(args.entities).words
'''
all_ents = Entities(args.entities).words

cats = {v : '' for k, v in all_ents.items()}
#ents = {k for k, v in all_ents.update(cats).items()}
ents = [k for k in cats.keys()]

#words = {'cats' : cats, 'ents' : ents}
#words = {'cats' : cats}
words = {'cats' : ['Face']}

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)

#for extraction_method in ['masked', 'unmasked']:
for extraction_method in ['unmasked']:

    ### Creating the folder for the word vectors
    out_folder = 'data/word_vectors_facets/bert_{}_prova'.format(extraction_method)
    os.makedirs(out_folder, exist_ok=True)

    for word_type, current_words in words.items():

        for current_word in tqdm(current_words):

            ### Preparing the file and path names

            file_current_word = re.sub(' ', '_', current_word)
            txt_file = '{}.txt'.format(file_current_word)

            if word_type == 'ents' and args.entities == 'wakeman_henson':
                short_folder = re.sub('\.', '', file_current_word)[:3].lower()
                short_folder = re.sub('[^a-zA-z0-9]', '_', short_folder)[:3]
                file_name = os.path.join('/import/cogsci/andrea/dataset/wexea_annotated_wiki/ready_corpus/final_articles', short_folder, txt_file)
            else:
                short_folder = current_word[:2]
                file_name = os.path.join('/import/cogsci/andrea/dataset/wikipedia_article_by_article', short_folder, txt_file)

            ### Extracting the list of sentences for the current word
            try:
                with open(file_name) as bert_txt:
                    #bert_lines = []
                    lines = [l for l in bert_txt.readlines()]
                    if word_type == 'ents' and args.entities == 'wakeman_henson':
                        mention = '[[{}|'.format(current_word)
                        selected_lines = [l.strip() for l in lines if mention in l]
                        if extraction_method == 'masked':
                            selected_lines = [l.replace(mention, '[MASK][[entity|') for l in selected_lines]
                        bert_lines = []
                        for line in selected_lines:
                            new_line = []
                            l_two = line.replace(']]', '[[')
                            l_three = [w for w in l_two.split('[[') if w != 'ANNOTATION']
                            l_four = [re.sub('\|\w+$', '', w) for w in l_three] 
                            l_five = [re.sub('^.+\|', '', w) for w in l_four]
                            if extraction_method == 'masked':
                                l_five = [w for w in l_five if w != current_word]
                            bert_lines.append(' '.join(l_five))
                    else:
                        common_noun = re.sub('_', ' ', current_word)
                        if '(' in common_noun:
                            common_noun = common_noun.split('(')[0].strip()
                        bert_lines = [l.strip() for l in lines if '{}'.format(common_noun) in l or '{}'.format(common_noun.lower()) in l]
                        if extraction_method == 'masked':
                            bert_lines = [l.replace(' {} '.format(common_noun.lower()), ' [MASK] ') for l in bert_lines]
                            bert_lines = [l.replace('{} '.format(common_noun), ' [MASK] ') for l in bert_lines]
                            bert_lines = [l.replace(' {}.'.format(common_noun.lower()), ' [MASK] ') for l in bert_lines]
                            bert_lines = [l.replace(' {},'.format(common_noun.lower()), ' [MASK] ') for l in bert_lines]

            except FileNotFoundError:
                print('impossible to extract the word vector for {}'.format(current_word))
                continue

            ### Extracting the BERT vectors
            
            bert_vectors = []        
            if len(bert_lines) > 20:
                bert_lines = bert_lines[:-10] 
            for ready_line in bert_lines:
                ready_line = ready_line.replace('\t', ' ')
                tokenized_input = bert_tokenizer.encode(ready_line)
                if len(tokenized_input) > 512:
                    tokenized_input = tokenized_input[:511] + [tokenized_input[-1]]
                input_ids = torch.tensor([tokenized_input])

                ### Making sure the sentence is correctly masked, in case
                if extraction_method == 'masked':
                    try:
                        assert len([i for i in tokenized_input if i == 103]) > 0
                    except AssertionError:
                        print('There a was a BERT tokenization mistake with sentence: {}'.format(ready_line))
                        continue

                ### Actually extracting the BERT vector
                with torch.no_grad():
                    try:
                        output = bert_model(input_ids)
                    except RuntimeError:
                        print('Error with {}'.format(ready_line))
                        pass
                    final_vector = output[2][12][0][-4:].numpy()
                    final_vector = numpy.average(final_vector, axis=0)
                    bert_vectors.append(final_vector)

            with open(os.path.join(out_folder, '{}.vec'.format(file_current_word)), 'w') as o:
                for vector in bert_vectors:
                    vector_to_txt(ready_line, vector, o)
                
