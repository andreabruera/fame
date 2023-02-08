from io_utils import ExperimentInfo, LoadEEG
from word_vector_enc_decoding.read_word_vectors import load_vectors_two

import argparse
import os
import random
import scipy
import sklearn

from scipy import stats
from sklearn import linear_model
from tqdm import tqdm

parser = argparse.ArgumentParser()

### Arguments having an effect on loading file
parser.add_argument('--analysis', type=str, \
                             ### Classification
                    choices=[\
                             ### Feature selection
                             'feature_selection', \
                             ### Experiment one classification
                             'classification_coarse', \
                             'classification_fine', \
                             'classification_common_proper', \

                             'whole_trial_classification_coarse', \
                             'whole_trial_classification_fine', \
                             'searchlight_whole_trial_classification_coarse', \
                             'searchlight_whole_trial_classification_fine', \
                             'searchlight_classification_coarse', \
                             'searchlight_classification_fine', \
                             'searchlight_classification_famous_familiar', \
                             'group_searchlight_classification_coarse', \
                             'group_searchlight_classification_fine', \
                             'group_searchlight_classification_famous_familiar', \
                             ### Experiment two classification
                             'classification_famous_familiar', \
                             ### Decoding & encoding
                             'decoding', \
                             'encoding', \
                             ### Searchlight
                             'rsa_searchlight', \
                             'feature_selection_group_searchlight', \
                             'group_searchlight',
                             ### Time resolved rsa
                             'time_resolved_rsa',
                             ], \
                    required=True, help='Indicates which \
                    analysis to run')
parser.add_argument('--corrected', \
                    action='store_true', \
                    default=False, help='Runs the classification \
                    controlling test samples for length')
parser.add_argument('--average', \
                    type=int, choices=list(range(25)), \
                    default=24, help='Defines whether \
                    to average ERPs and how many at a time')
parser.add_argument('--subsample', choices=['subsample',
                         'subsample_2', 'subsample_3', 'subsample_4',\
                         'subsample_6', 'subsample_8', 'subsample_10', \
                         'subsample_12', 'subsample_14', 'subsample_16', \
                         'subsample_18', 'subsample_20', 'subsample_22', \
                         'subsample_24', 'subsample_26', \
                         'all_time_points'], \
                    required=True, help='Defines whether \
                    to subsample by averaging sample within 40ms window')
parser.add_argument('--data_kind', choices=['erp', 'combined',
                                            'alpha', 'beta', 'lower_gamma', 
                                            'higher_gamma', 'delta', 'theta',
                                            ], \
                    required=True, help='Time-frequency or ERP analyses?') 

### Arguments which have an effect on folder structure
parser.add_argument('--entities', type=str,  \
                    choices=[\
                    ### Experiment one
                    'individuals_only', 'individuals_and_categories', 
                    'all_to_individuals', 'all_to_categories', 
                    'individuals_to_categories', 'categories_only',
                    'people_only', 'places_only',
                    ], required=True, \
                    help='Restricts the analyses to individuals only ')
parser.add_argument('--semantic_category', choices=['people', 'places', 
                                                    'famous', 'familiar',
                                                    'all',
                                                    ], \
                    required=True, help='Which semantic category?') 
### Arguments which do not affect output folder structure
parser.add_argument('--data_folder', \
                    type=str, \
                    required=True, help='Indicates where \
                    the experiment files are stored')
parser.add_argument('--ceiling', \
                    action='store_true', \
                    default=False, help=\
                    'Ceiling instead of model?')
parser.add_argument('--feature_selection_method', required=False, \
                    type=str, help='Feature selection?', \
                    choices=['fisher', 'stability', 'distinctiveness', 'noisiness', 'mutual_information', \
                            'feature_selection_group_searchlight', 'none', 'attribute_correlation'])

parser.add_argument('--experiment_id', \
                    choices=['one', 'two', 'pilot'], \
                    required=True, help='Indicates which \
                    experiment to run')

### Enc-decoding specific
parser.add_argument('--word_vectors', required=False, \
                    choices=[\
                             'orthography',
                             'imageability',
                             'familiarity',
                             'word_length',
                             'frequency',
                             'log_frequency',
                             ### English
                             # Contextualized
                             'BERT_base_en_sentence', 'BERT_large_en_sentence',\
                             'BERT_base_en_mentions', 'BERT_large_en_mentions',\
                             'ELMO_small_en_mentions', 'ELMO_original_en_mentions',\
                             'BERT_large',
                             # Contextualized + knowledge-aware
                             'BERT_base_en_sentence', 'BERT_large_en_sentence',\
                             'LUKE_base_en_mentions', 'LUKE_large_en_mentions',\
                             # Static
                             'w2v', \
                             # Static + knowledge-aware
                             'wikipedia2vec', \
                             # Knowledge-only
                             'transe', 
                             ### Italian
                             'gpt2-xl',
                             'gpt2-large',
                             'xlm-roberta-large',
                             'LUKE_large',
                             'SPANBERT_large', \
                             'MBERT', \
                             'ITBERT',
                             'it_w2v', \
                             'it_wikipedia2vec', \
                             'ITGPT2medium',
                             'BERT_base_it_mentions', \
                             ### Ceiling
                             'ceiling',
                             ### Category
                             'coarse_category',
                             'famous_familiar',
                             'fine_category',
                             'mixed_category',
                             ], \
                    help='Which computational model to use for decoding?')
args = parser.parse_args()

out_folder = os.path.join('clustered_sentence_selection', 
                          args.experiment_id, 
                          args.word_vectors)
os.makedirs(out_folder, exist_ok=True)

for n in range(1, 34):
    print('Subject {}'.format(n))
    experiment = ExperimentInfo(args, subject=n)
    ### Preparing the vectors
    comp_vectors = load_vectors_two(args, experiment, n, clustered=True)
    for k, v in comp_vectors.items():
        assert len(v) <= 50
    correlations = {k : {i : 0 for i in range(50)} for k in comp_vectors.keys()}
    ### Preparing the eeg_data
    eeg_data = LoadEEG(args, experiment, n)
    ### Flattening
    eeg = {experiment.trigger_to_info[k][0] : v.flatten() for k, v in eeg_data.data_dict.items()}
    ### Preparing the data
    input_target = [(eeg[k], vec, k, vec_i) for k, v in comp_vectors.items() for vec_i, vec in enumerate(v) if k in eeg.keys()]
    input_target = random.sample(input_target, k=len(input_target))
    for d_i, test_tuple in tqdm(enumerate(input_target)):
        train_data = [k for k_i, k in enumerate(input_target) if k_i!=d_i]
        train_input = [k[0] for k in train_data]
        train_target = [k[1] for k in train_data]
        model = sklearn.linear_model.Ridge().fit(train_input, train_target)
        prediction = model.predict([test_tuple[0]])
        assert len(prediction) == 1
        corr = scipy.stats.pearsonr(prediction[0], test_tuple[1])[0]
        correlations[test_tuple[2]][test_tuple[3]] += corr
    correlations = {k : sorted(v.items(), reverse=True, key=lambda item : item[1]) for k, v in correlations.items()}
    with open(os.path.join(out_folder, 'sub-{:02}.selection'.format(n)), 'w') as o:
        for k, v in correlations.items():
            o.write('{}\t'.format(k))
            for idx, corr in v:
                o.write('{}, {}\t'.format(idx, corr))
            o.write('\n')
