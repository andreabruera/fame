import argparse
import itertools
import logging
import multiprocessing
import numpy
import os
import random
import scipy
import sklearn

from tqdm import tqdm

from feature_selection.feature_selection import compute_and_write
from feature_selection.scores import get_scores
from io_utils import ExperimentInfo, LoadEEG, prepare_folder, tfr_frequencies
from plot_scripts.plot_classification import plot_classification
from plot_scripts.plot_feature_selection_comparison import plot_feature_selection_comparison
from plot_scripts.plot_decoding_results_breakdown import plot_decoding_results_breakdown
from plot_scripts.plot_decoding_scores_comparison import plot_decoding_scores_comparison
from searchlight.searchlight_utils import SearchlightClusters, join_searchlight_results, run_searchlight, write_plot_searchlight
from searchlight.group_searchlight import group_searchlight
from searchlight.feature_selection_group_searchlight import feature_selection_group_searchlight
from word_vector_enc_decoding.read_word_vectors import WordVectors, load_vectors_two
from word_vector_enc_decoding.encoding_decoding_utils import prepare_and_test, write_enc_decoding
from classification.time_resolved_rsa import time_resolved_rsa

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

#numpy.seterr(all='raise')
#numbers = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000, 28000, ]
numbers = [\
           5, 50, 100, 200, 250, 300, 400, 500, 600, 700, 800, 900, \
           2000, 3000, 5000, 6000, 8000, 9000, \
           1000, 4000, 7000, 10000, 11000, 12000, \
           13000, 16000, 19000, 22000, 25000, 28000, \
           1000, 2500, 5000, 7500, 10000, 12500, 15000, \
           17500, 20000, 22500, 25000, 27500, 30000,
           50000, 100000]
#modes = ['aggregate', 'stability', 'noisiness', 'distinctiveness', 'stability_searchlight', 'rsa']
#choices = ['{}{}'.format(mode, number) for number in numbers for mode in modes]
#choices = choices + ['no_reduction', 'rsa']
choices = [str(n) for n in numbers] + ['no_reduction']

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

parser.add_argument('--data_kind', choices=[

                        'erp', 'combined',
                        'alpha', 'beta', 'lower_gamma', 
                        'higher_gamma', 'delta', 'theta',
                        ### ATLs
                        'bilateral_anterior_temporal_lobe', 'left_atl', 'right_atl', 
                        ### Lobes
                        'left_frontal_lobe', 'right_frontal_lobe', 'bilateral_frontal_lobe',
                        'left_temporal_lobe', 'right_temporal_lobe', 'bilateral_temporal_lobe',
                        'left_parietal_lobe', 'right_parietal_lobe', 'bilateral_parietal_lobe',
                        'left_occipital_lobe', 'right_occipital_lobe', 'bilateral_occipital_lobe',
                        'left_limbic_system', 'right_limbic_system', 'bilateral_limbic_system',
                        ### Networks
                        'language_network', 'general_semantics_network',
                        'default_mode_network', 'social_network', 
                                            ], 
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
parser.add_argument('--feature_reduction', \
                    choices=choices, default='no_reduction')

### Arguments which do not affect output folder structure
parser.add_argument('--data_folder', \
                    type=str, \
                    required=True, help='Indicates where \
                    the experiment files are stored')
parser.add_argument('--debugging', \
                    action='store_true', \
                    default=False, help='Runs the classification \
                    without multiprocessing so as to allow debugging')
parser.add_argument('--corrected', \
                    action='store_true', \
                    default=False, help='Runs the classification \
                    controlling test samples for length')
parser.add_argument('--plot', \
                    action='store_true', \
                    default=False, help=\
                    'Plotting instead of testing?')
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
                             'gpt2',
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
parser.add_argument('--evaluation_method', required=False, \
                    choices=['pairwise', 'ranking'], \
                    help='Which evaluation method to use for decoding?')
'''
parser.add_argument('--time_resolved', \
                    action='store_true', \
                    default=True, help='Runs the analyses on a time-by-time \
                    basis')
parser.add_argument('--training_split', type=int, \
                    default=90, choices=[80, 90], help='Percentage of data \
                    used to train')
### Obsolete arguments
parser.add_argument('--test_length', \
                    action='store_true', \
                    default=False, help='Testing for length effect?')
'''
parser.add_argument('--wv_dim_reduction', \
                    default='no_dim_reduction', \
                    choices=['no_dim_reduction', 'pca50', 'pca60', 'pca70', \
                             'pca80', 'pca90', 'pca99'], help='Defines whether \
                    to employ dimensionality on word vectors or not')
args = parser.parse_args()

searchlight_distance = 22

if args.feature_reduction != 'no_reduction':
    args.feature_reduction = int(args.feature_reduction)

if 'coarse' in args.analysis and \
           args.semantic_category in ['people', 'places'] and \
           args.experiment_id == 'one':
    raise RuntimeError('Wrong data split - impossible to run a coarse'\
                       ' classification task on people or places exclusively')
if args.semantic_category in ['famous', 'familiar'] and \
           args.experiment_id == 'one':
    raise RuntimeError('Wrong data split - there is no famous/familiar distintion'\
                       'for experiment one!')

#if args.experiment_id == 'two':
#    from classification.classification_utils_two import run_time_resolved_classification, run_searchlight_classification, run_splitting
#if args.experiment_id == 'one' and 'classification' in args.analysis:
#    raise RuntimeError('This has to be corrected! - see function run_time_resolved_classification in experiment two')
#    from classification.classification_utils_one import run_classification
from classification.classification_utils_two import run_time_resolved_classification, run_searchlight_classification 

### Plotting
if args.plot:
    if 'whole_trial' in args.analysis:
        plot_feature_selection_comparison(args, experiment)
    elif args.analysis == 'time_resolved_rsa':
        plot_classification(args)
    elif 'coding' in args.analysis:
        if args.experiment_id == 'one':
            plot_decoding_results_breakdown(args)
            plot_feature_selection_comparison(args)
            plot_decoding_scores_comparison(args)
        else:
            plot_decoding_results_breakdown(args)
            raise RuntimeError('Still to be implemented!')
    elif 'searchlight' in args.analysis:
        group_searchlight(args)
    elif 'classification' in args.analysis:
        #if 'searchlight' in args.analysis:
        #    for electrode in tqdm(range(128)):
        #        plot_classification(args, electrode)
        #else:
        #    plot_classification(args)
        plot_classification(args)

### Getting the results
else:
    #frequencies = tfr_frequencies(args)
    #n_subjects = experiment.subjects
    processes = int(os.cpu_count()/8)
    #processes = int((os.cpu_count()/3)*2)

    if __name__ == '__main__':

        ### Feature selection only
        if args.analysis == 'feature_selection':
            
            out_path = prepare_folder(args)\
                       .replace('feature_selection', 'feature_scores')\
                       .replace('average_1', 'average_0')
            out_path = out_path.replace('/{}'.format(args.wv_dim_reduction), '')
            os.makedirs(out_path, exist_ok=True)

            ### Various stability measures
            if 'searchlight' not in args.feature_selection_method:
                data_paths = experiment.eeg_paths
                trigger_to_info = experiment.trigger_to_info

                all_args = [[n, data_paths[n], args, \
                             trigger_to_info, experiment] \
                             for n in range(n_subjects)]

                logging.info('Now starting feature extraction')

                if args.debugging:
                    feature_maps = list()
                    for arg in all_args:
                        feature_maps.append(compute_and_write(arg))
                else:
                    with multiprocessing.Pool(processes=processes) as p:
                        feature_maps = p.map(compute_and_write, all_args)
                    p.close()
                    p.join()

                feature_maps = numpy.array(feature_maps)
                assert feature_maps.shape[0] == 33

                for current_sub in range(1, 34):

                    other_maps = numpy.delete(feature_maps.copy(), \
                                              current_sub, axis=0)
                    assert other_maps.shape[0] == 32
                    other_maps = numpy.average(other_maps, axis=0)
                    if args.feature_selection_method == 'attribute_correlation':
                        file_path = os.path.join(out_path, \
                                    'sub-{:02}_feature_selection.{}_{}'\
                                    .format(current_sub, args.word_vectors, args.feature_selection_method)\
                                    )
                    else:
                        file_path = os.path.join(out_path, \
                                    'sub-{:02}_feature_selection.{}'\
                                    .format(current_sub, args.feature_selection_method)\
                                    )
                    with open(file_path, 'w') as o:
                        for i in other_maps:
                            #for j in other_maps[i, :]:
                            o.write('{}\t'.format(i))
                        o.write('\n')

            ### Group searchlight feature selection
            elif args.feature_selection_method == 'feature_selection_group_searchlight':
                data_path = experiment.eeg_paths[1]
                eeg_data = LoadEEG(args, data_path, 1, experiment)
                all_times = eeg_data.all_times

                #if args.data_kind == 'erp':
                if args.data_kind != 'time_frequency':
                    score_collector = feature_selection_group_searchlight(args, experiment)
                    ### TODO: turn it into a regular-sized array
                    assert score_collector.shape[0] == 33 \
                            and score_collector.shape[1] == 128
                    erp_collector = list()
                    for sub_erp in score_collector:
                        time_filtered_array = from_clusters_to_original_shape(args, sub_erp, all_times)
                        erp_collector.append(time_filtered_array)
                    erp_collector = numpy.array(erp_collector)
                    assert score_collector.shape[0] == 33 \
                            and score_collector.shape[1] == 128
                    for n_one, t_stats in enumerate(erp_collector):
                        file_path = os.path.join(out_path, \
                            'sub-{:02}_{}_feature_selection.{}_searchlight'\
                            .format(n_one, args.word_vectors, \
                            args.data_kind))
                        t_stats = t_stats.flatten()
                        with open(file_path, 'w') as o:
                            for j in t_stats:
                                o.write('{}\t'.format(j))
                            o.write('\n')
                            '''
                            for i in range(t_stats.shape[0]):
                                for j in t_stats[i, :]:
                                    o.write('{}\t'.format(j))
                                o.write('\n')
                            '''

                elif args.data_kind == 'time_frequency':
                    #score_collector = {k : list() for k in range(33)}
                    score_collector = list()
                    for f in frequencies:
                        freq_collector = feature_selection_group_searchlight(args, experiment, hz='{}hz_'.format(f))
                        ### Cheeky dump
                        #freq_collector.dump('{}.npy'.format(f))
                        #freq_collector = numpy.load('{}.npy'.format(f), allow_pickle=True)
                        ### TODO: turn it into a regular-sized array
                        score_collector.append(freq_collector)
                    score_collector = numpy.array(score_collector)
                    # Frequency x electrodes
                    # Electrodes x subjects
                    score_collector = score_collector.\
                                      swapaxes(0, 2).\
                                      swapaxes(0, 1)
                    assert score_collector.shape[0] == 33 \
                       and score_collector.shape[1] == 128 \
                       and score_collector.shape[2] == len(frequencies)

                    tfr_collector = list()
                    for f_i in range(len(frequencies)):
                        freq_collector = list()
                        for sub_tfr in score_collector:
                            time_filtered_array = from_clusters_to_original_shape(args, sub_tfr[:,f_i, :], all_times)
                            freq_collector.append(time_filtered_array)
                        tfr_collector.append(freq_collector)
                    tfr_collector = numpy.array(tfr_collector)
                    ### TODO: turn each of the score collector arrays into its appropriate shape
                    # Frequency x electrodes
                    # Electrodes x subjects
                    tfr_collector = tfr_collector\
                                      .swapaxes(0, 2)\
                                      .swapaxes(0, 1)
                    assert tfr_collector.shape[0] == 33 \
                       and tfr_collector.shape[1] == 128 \
                       and tfr_collector.shape[2] == len(frequencies)

                    for n_one, t_stats in enumerate(tfr_collector):
                        file_path = os.path.join(out_path, \
                                'sub-{:02}_{}_feature_selection.{}_searchlight'\
                                .format(n_one, args.word_vectors, \
                                args.data_kind))
                        print('\n\n{}\n\n'.format(file_path))
                        t_stats = t_stats.flatten()
                        with open(file_path, 'w') as o:
                            for j in t_stats:
                                o.write('{}\t'.format(j))
                            o.write('\n')
                        '''
                        with open(file_path, 'w') as o:
                            for i in range(t_stats.shape[0]):
                                for j in t_stats[i, :]:
                                    o.write('{}\t'.format(j))
                                o.write('\n')
                        '''

        ### Classification, decoding or searchlight

        elif args.analysis == 'time_resolved_rsa':
            ### Group searchlight analysis
            if args.plot:
                group_searchlight(args)
            else:
                if not args.word_vectors:
                    raise RuntimeError('You need to specify the word vector model for encoding/decoding')
                if args.debugging:
                    for n in range(1, 34):
                        time_resolved_rsa([args, n, False])
                else:
                    with multiprocessing.Pool(processes=processes) as pool:
                        pool.map(time_resolved_rsa, [(args, n, False) for n in range(1, 34)])
                        pool.close()
                        pool.join()
        elif args.analysis == 'rsa_searchlight':
            if not args.word_vectors:
                raise RuntimeError('You need to specify the word vector model for encoding/decoding')
            if args.debugging:
                for n in range(1, 34):
                    time_resolved_rsa([args, n, True])
            else:
                with multiprocessing.Pool(processes=processes) as pool:
                    pool.map(time_resolved_rsa, [(args, n, True) for n in range(1, 34)])
                    pool.close()
                    pool.join()

        ### Encoding / decoding
        elif args.analysis in ['encoding', 'decoding']:
            print(args.evaluation_method, args.word_vectors)
            ### Checks
            if not args.evaluation_method and not args.word_vectors:
                raise RuntimeError('You need to specify both evaluation method '\
                                   'and the word vector model for encoding/decoding')
            ### Loading word vectors
            experiment = ExperimentInfo(args)
            #if args.experiment_id == 'one' and args.word_vectors != 'ceiling':
            #    comp_model = WordVectors(args, experiment)
            #    comp_vectors = comp_model.vectors
            #if args.experiment_id == 'two':
            #    raise RuntimeError('still not implemented')

            accuracies = list()
            word_by_word = dict()
            ### Loading all EEG data
            if args.word_vectors == 'ceiling':
                ceiling = dict()
                for n_ceiling in range(1, 34):
                    eeg_data_ceiling = LoadEEG(args, experiment, n_ceiling)
                    eeg_ceiling = eeg_data_ceiling.data_dict
                    for k, v in eeg_ceiling.items():
                        ### Adding and flattening
                        if k not in ceiling.keys():
                            ceiling[k] = list()
                        ceiling[k].append(v.flatten())
                #ceiling = {experiment.trigger_to_info[k][0] : v for k, v in ceiling.items()}
            else:
                if args.experiment_id == 'one':
                    experiment = ExperimentInfo(args, 
                                                )
                    comp_vectors = load_vectors_two(args, experiment, 33)

            #for n in range(n_subjects):
            for n in range(1, 34):
                if args.experiment_id == 'two':
                    experiment = ExperimentInfo(args, 
                                                subject=n
                                                )
                print('Subject {}'.format(n))
                if args.word_vectors == 'ceiling':
                    comp_vectors = {k : numpy.average([vec for vec_i, vec in enumerate(v) if vec_i!=n], axis=0) for k, v in ceiling.items()}
                else:
                    if args.experiment_id == 'two':
                        comp_vectors = load_vectors_two(args, experiment, n)

                sub_accuracy, word_by_word_evaluation, \
                        times = prepare_and_test(n, args, experiment, \
                                     comp_vectors)
                accuracies.append(sub_accuracy)

                ### Collecting performances on a word by word basis
                #if n == 0:
                for w, v in word_by_word_evaluation.items():
                    if w not in word_by_word.keys():
                        word_by_word[w] = [v]
                    else:
                        word_by_word[w].append(v)

            write_enc_decoding(args, accuracies, word_by_word)

        ### Classification

        elif 'classification' in args.analysis and 'group' not in args.analysis:

            ### Time-resolved classification
            if not 'searchlight' in args.analysis:

                ### Single subject, for debugging
                if args.debugging:

                    for n in range(1, 34):
                        print('Subject {}'.format(n))
                        run_time_resolved_classification([args, n])
                
                ### All subjects with a separate process, much quicker
                else:

                    with multiprocessing.Pool(processes=processes) as pool:
                        sub_scores = pool.map(run_time_resolved_classification, [(args, 
                                                n,
                                                ) for n in range(1, 34)])
                        pool.close()
                        pool.join()

            ### Searchlight classification
            else:
                
                searchlight_clusters = SearchlightClusters(max_distance=searchlight_distance)

                for n in range(1, 34):

                    ### Preparing clusters
                    experiment = ExperimentInfo(args, subject=n)
                    eeg = LoadEEG(args, experiment, n) 

                    #step = 8
                    step = 26
                    relevant_times = [t_i for t_i, t in enumerate(eeg.times) if t_i+step<len(eeg.times)][::step]
                    #relevant_times = [t_i for t_i, t in enumerate(eeg.times) if t_i+(step*2)<len(eeg.times)][::step]
                    #explicit_times = [eeg.times[t] for t in relevant_times]
                    explicit_times = [eeg.times[t+int(step/2)] for t in relevant_times]

                    electrode_indices = [searchlight_clusters.neighbors[center] for center in range(128)]
                    clusters = [(e_s, t_s) for e_s in electrode_indices for t_s in relevant_times]

                    print('Searchlight subject n. {}'.format(n))
                    if args.debugging:
                        results = list()
                        for cluster in clusters:
                            results.append(run_searchlight_classification([ 
                                                args, experiment, eeg, cluster, step]))
                    else:
                        with multiprocessing.Pool(processes=processes) as p:
                            results = p.map(run_searchlight_classification, 
                            [[args, experiment, eeg, cluster, step] for cluster in clusters])
                            p.terminate()
                            p.join()

                    results_array = join_searchlight_results(results, relevant_times)
                    write_plot_searchlight(args, n, explicit_times, results_array)

