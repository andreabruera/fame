import argparse
import numpy
import scipy
import sklearn

from scipy import stats
from sklearn.preprocessing import LabelEncoder

def read_args():

    parser = argparse.ArgumentParser()

    ### plotting
    parser.add_argument(
                        '--plot', 
                        action='store_true', 
                        default=False, 
                        help='Plotting instead of testing?'
                        )

    ### applying to anything
    parser.add_argument(
                        '--experiment_id',
                        choices=['one', 'two', 'pilot'], 
                        required=True,
                        help='Which experiment?'
                        )
    parser.add_argument(
                        '--average', 
                        type=int, choices=list(range(25)), 
                        default=24, 
                        help='How many ERPs to average?'
                        )
    parser.add_argument('--analysis', 
                        type=str,
                        choices=[
                                 'time_resolved_rsa_encoding',
                                 ### Experiment one classification
                                 'classification_coarse', 
                                 'classification_fine', 
                                 'classification_common_proper', 

                                 'searchlight_classification_coarse',
                                 'searchlight_classification_fine',
                                 'searchlight_classification_famous_familiar',
                                 ### Experiment two classification
                                 'classification_famous_familiar',
                                 ### Decoding & encoding
                                 'decoding',
                                 'encoding',
                                 ### Searchlight
                                 'rsa_searchlight',
                                 ### Time resolved rsa
                                 'time_resolved_rsa',
                                 ### temporal generalization
                                 'temporal_generalization'
                                 ],
                        required=True, help='Which analysis?'
                        )
    parser.add_argument('--temporal_resolution', choices=[4, 5, 10, 25, 50, 100],
                        type=int, required=True)
    '''
    parser.add_argument(
                        '--subsample', choices=['subsample',
                             'subsample_2', 'subsample_3', 'subsample_4',\
                             'subsample_5', 'subsample_7', 'subsample_9',\
                             'subsample_6', 'subsample_8', 'subsample_10', \
                             'subsample_12', 'subsample_14', 'subsample_16', \
                             'subsample_18', 'subsample_20', 'subsample_22', \
                             'subsample_24', 'subsample_26', \
                             'all_time_points'], \
                        required=True, help='Defines whether \
                        to subsample by averaging sample within 40ms window')
    '''

    parser.add_argument('--data_kind', choices=[
                            'erp', 
                            'alpha', 
                            'beta', 
                            'lower_gamma', 
                            'higher_gamma', 
                            'delta', 
                            'theta',
                            ### ATLs
                            'bilateral_anterior_temporal_lobe', 
                            'left_atl', 
                            'right_atl', 
                            ### Lobes
                            'left_frontal_lobe', 'right_frontal_lobe', 'bilateral_frontal_lobe',
                            'left_temporal_lobe', 'right_temporal_lobe', 'bilateral_temporal_lobe',
                            'left_parietal_lobe', 'right_parietal_lobe', 'bilateral_parietal_lobe',
                            'left_occipital_lobe', 'right_occipital_lobe', 'bilateral_occipital_lobe',
                            'left_limbic_system', 'right_limbic_system', 'bilateral_limbic_system',
                            ### Networks
                            'language_network', 
                            'general_semantics_network',
                            'default_mode_network', 
                            'social_network', 
                            ], 
                        required=True, 
                        help='Time-frequency, ERP or source analyses?'
                        ) 
    ### Arguments which do not affect output folder structure
    parser.add_argument(
                        '--data_folder',
                        type=str,
                        required=True, 
                        help='Indicates where the experiment files are stored',
                        )
    parser.add_argument(
                        '--debugging',
                        action='store_true',
                        default=False, 
                        help='No multiprocessing')

    ### experiment-specific arguments
    parser.add_argument('--entities', 
                        type=str,
                        choices=[
                        ### Experiment one
                        'individuals_only', 
                        'individuals_and_categories', 
                        'all_to_individuals', 
                        'all_to_categories', 
                        'individuals_to_categories', 
                        'categories_only',
                        'people_only', 
                        'places_only',
                        ], 
                        required=True,
                        help='Which types of entities to use?'
                        )
    parser.add_argument(
                        '--semantic_category', 
                        choices=[
                                 'people', 
                                 'places', 
                                 'famous', 
                                 'familiar',
                                 'all',
                                 ],
                        required=True, 
                        help='Which semantic category?'
                        ) 
    parser.add_argument(
                        '--corrected',
                        action='store_true',
                        default=False, 
                        help='Controlling test samples for length?')
    ### Enc-decoding specific
    parser.add_argument(
                        '--word_vectors',
                        required=False,
                        choices=[
                                 'ceiling',
                                 'orthography',
                                 'imageability',
                                 'familiarity',
                                 'word_length',
                                 'frequency',
                                 'log_frequency',
                                 ### English
                                 # Contextualized
                                 'BERT_base_en_sentence', 'BERT_large_en_sentence',
                                 'BERT_base_en_mentions', 'BERT_large_en_mentions',
                                 'ELMO_small_en_mentions', 'ELMO_original_en_mentions',
                                 'BERT_large',
                                 'gpt2',
                                 # Contextualized + knowledge-aware
                                 'BERT_base_en_sentence', 'BERT_large_en_sentence',
                                 'LUKE_base_en_mentions', 'LUKE_large_en_mentions',
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
                                 'SPANBERT_large', 
                                 'MBERT', 
                                 'ITBERT',
                                 'it_w2v',
                                 'it_wikipedia2vec',
                                 'ITGPT2medium',
                                 'BERT_base_it_mentions',
                                 ### Ceiling
                                 'ceiling',
                                 ### Category
                                 'coarse_category',
                                 'famous_familiar',
                                 'fine_category',
                                 'mixed_category',
                                 ],
                        help='Which computational model to use for decoding?'
                        )
    parser.add_argument(
                        '--evaluation_method', 
                        default='pairwise',
                        choices=['pairwise', 'ranking'],
                        help='Which evaluation method to use for decoding?'
                        )
    parser.add_argument(
                        '--searchlight_spatial_radius', 
                        choices=[
                                 ### 30mm radius, used in
                                 ### Collins et al. 2018, NeuroImage, Distinct neural processes for the perception of familiar versus unfamiliar faces along the visual hierarchy revealed by EEG
                                 ### Su et al., Optimising Searchlight Representational Similarity Analysis (RSA) for EMEG
                                 'large_distance',
                                 ### 20mm radius, used in 
                                 ### Su et al. 2014, Mapping tonotopic organization in human temporal cortex: representational similarity analysis in EMEG source space. Frontiers in neuroscience
                                 'small_distance',
                                 ### 5 closest electrodes
                                 ### Graumann et al. 2022, The spatiotemporal neural dynamics of object location representations in the human brain, nature human behaviour
                                 'fixed'
                                 ], 
                        required=True
                        )
    parser.add_argument(
                        '--searchlight_temporal_radius', 
                        choices=[
                                 ### 100ms radius
                                 'large',
                                 'medium',
                                 ### 50ms radius
                                 'small',
                                 ], 
                        required=True
                        )

    args = parser.parse_args()

    check_args(args)
    
    return args

def check_args(args):
    ### checking inconsistencies in the args
    marker = False
    ### experiment one
    if args.experiment_id == 'one':
        if 'coarse' in args.analysis and args.semantic_category in ['people', 'places']:
            marker = True
        if args.semantic_category in ['famous', 'familiar']:
            marker = True

def split_train_test(args, split, eeg, experiment, comp_vectors):

    ### Selecting the relevant index for the trigger_to_info dictionary
    if 'coarse' in args.analysis:
        cat_index = 1
    elif 'fine' in args.analysis or 'famous' in args.analysis:
        cat_index = 2

    test_samples = list()
    test_true = list()

    for trig in split:
        if args.analysis == 'time_resolved_rsa_encoding':
            trig = experiment.trigger_to_info[trig][0]
            vecs = comp_vectors[trig]
        erps = eeg[trig]
        if not isinstance(erps, list):
            erps = [erps]
            if args.analysis == 'time_resolved_rsa_encoding':
                vecs = [vecs]
        if args.analysis == 'time_resolved_rsa_encoding':
            test_samples.extend(vecs)
            test_true.extend(erps)
        else:
            test_samples.extend(erps)
            test_true.extend([experiment.trigger_to_info[trig][cat_index] for i in range(len(erps))])

    train_samples = list()
    train_true = list()

    for k, erps in eeg.items():
        if args.analysis == 'time_resolved_rsa_encoding':
            k = {v[0] : k for k, v in experiment.trigger_to_info.items()}[k]
        if not isinstance(erps, list):
            erps = [erps]
        if k not in split:
            if args.analysis == 'time_resolved_rsa_encoding':
                k = experiment.trigger_to_info[k][0]
                vecs = [comp_vectors[k]]
                train_samples.extend(vecs)
                train_true.extend(erps)
            else:
                train_samples.extend(erps)
                train_true.extend([experiment.trigger_to_info[k][cat_index] for i in range(len(erps))])

    if args.analysis != 'time_resolved_rsa_encoding':
        ### Check labels
        if args.experiment_id == 'two':
            assert len(list(set(train_true))) == 2
            assert len(list(set(test_true))) == 2
            if 'coarse' in args.analysis:
                assert 'person' in train_true
                assert 'place' in train_true
                assert 'person' in test_true
                assert 'place' in test_true
            if 'famous' in args.analysis:
                assert 'famous' in train_true
                assert 'familiar' in train_true
                assert 'famous' in test_true
                assert 'familiar' in test_true
        elif args.experiment_id == 'one':
            if 'coarse' in args.analysis:
                assert len(list(set(train_true))) == 2
                assert len(list(set(test_true))) == 2
                assert 'person' in train_true
                assert 'place' in train_true
                assert 'person' in test_true
                assert 'place' in test_true
            if 'fine' in args.analysis:
                if args.semantic_category == 'people':
                    assert len(list(set(train_true))) == 4
                if args.semantic_category == 'places':
                    assert len(list(set(train_true))) == 4
                if args.semantic_category == 'all':
                    assert len(list(set(train_true))) == 8

        train_true = LabelEncoder().fit_transform(train_true)
        test_true = LabelEncoder().fit_transform(test_true)
        
        #train_labels = len(list(set(train_true)))
    else:
        train_true = numpy.array(train_true, dtype=numpy.float64)
        test_true = numpy.array(test_true, dtype=numpy.float64)
        test_samples = numpy.array(test_samples, dtype=numpy.float64)
        train_samples = numpy.array(train_samples, dtype=numpy.float64)

        assert train_samples.shape[0] == len(eeg.keys())-2
        assert len(test_samples) == 2

    return train_true, test_true, train_samples, test_samples

def evaluate_pairwise(args, train_true, test_true, train_samples, test_samples):
    ### orthography uses a unique distance metric
    #if args.word_vectors == 'orthography':
    #    test_samples = [[levenshtein(tst, tr) for tr in train_samples] for tst in test_samples]
    #else:
    ### single values use absolute distance
    #if type(test_samples[0]) == numpy.float64:
    if type(test_samples[0]) in [int, float, numpy.float64]:
        ### for categories, the higher the value the more similar, so we invert
        #if set(test_samples.tolist()) == set([0, 1]):
        #    test_samples = [[1 - abs(tst-tr) for tr in train_samples] for tst in test_samples]
        #else:
        #    test_samples = [[-abs(tst-tr) for tr in train_samples] for tst in test_samples]
        test_samples = [[1 - abs(tst-tr) for tr in train_samples] for tst in test_samples]

    ### for vectors, we use correlation
    ### NB: the higher the value, the more similar the items!
    else:
        test_samples = [[scipy.stats.pearsonr(tst, tr)[0] for tr in train_samples] for tst in test_samples]
    ### Encoding targets (brain images)
    test_samples = [numpy.sum([t*corrs[t_i] for t_i, t in enumerate(train_true)], axis=0) for corrs in test_samples]

    wrong = 0.
    for idx_one, idx_two in [(0, 1), (1, 0)]:
        wrong += scipy.stats.pearsonr(test_samples[idx_one], test_true[idx_two])[0]
    correct = 0.
    for idx_one, idx_two in [(0, 0), (1, 1)]:
        correct += scipy.stats.pearsonr(test_samples[idx_one], test_true[idx_two])[0]

    if correct > wrong:
        #accuracies.append(1)
        accuracy = 1
    else:
        #accuracies.append(0)
        accuracy = 0

    return accuracy

