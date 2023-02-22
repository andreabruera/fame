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

from general_utils import read_args
from io_utils import ExperimentInfo, LoadEEG, prepare_folder, tfr_frequencies

from plot_scripts.plot_classification import plot_classification
from plot_scripts.plot_decoding_results_breakdown import plot_decoding_results_breakdown
from plot_scripts.plot_decoding_scores_comparison import plot_decoding_scores_comparison

from searchlight.searchlight_utils import SearchlightClusters, join_searchlight_results, write_plot_searchlight
from searchlight.group_searchlight import group_searchlight

from word_vector_enc_decoding.read_word_vectors import WordVectors, load_vectors_two
from word_vector_enc_decoding.encoding_decoding_utils import prepare_and_test, write_enc_decoding

from classification.time_resolved_rsa import time_resolved_rsa
from classification.classification_utils_two import run_time_resolved_classification, run_searchlight_classification 

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

#numpy.seterr(all='raise')

args = read_args()

searchlight_clusters = SearchlightClusters(args)

### Plotting

if args.plot:
    if 'whole_trial' in args.analysis:
        plot_feature_selection_comparison(args, experiment)
    elif args.analysis in ['time_resolved_rsa', 'time_resolved_rsa_encoding']:
        plot_classification(args)
    elif 'coding' in args.analysis:
        if args.experiment_id == 'one':
            plot_decoding_results_breakdown(args)
            plot_feature_selection_comparison(args)
            plot_decoding_scores_comparison(args)
        else:
            plot_decoding_results_breakdown(args)
    elif 'searchlight' in args.analysis:
        group_searchlight(args)
    elif 'classification' in args.analysis:
        plot_classification(args)

### Getting the results
else:
    processes = int(os.cpu_count()/8)

    experiment = ExperimentInfo(args)

    if __name__ == '__main__':

        ### time-resolved rsa
        if args.analysis in ['time_resolved_rsa', 'time_resolved_rsa_encoding']:
            ### Group searchlight analysis
            if not args.word_vectors:
                raise RuntimeError('You need to specify the word vector model for encoding/decoding')
            if args.debugging:
                for n in range(1, experiment.subjects+1):
                    time_resolved_rsa([args, n, False, searchlight_clusters])
            else:
                with multiprocessing.Pool(processes=processes) as pool:
                    pool.map(time_resolved_rsa, [(args, n, False, searchlight_clusters) for n in range(1, experiment.subjects+1)])
                    pool.close()
                    pool.join()

        ### rsa searchlight in the range 100-750ms
        elif args.analysis == 'rsa_searchlight':
            if not args.word_vectors:
                raise RuntimeError('You need to specify the word vector model for encoding/decoding')
            if args.debugging:
                for n in range(1, experiment.subjects+1):
                    time_resolved_rsa([args, n, True, searchlight_clusters])
            else:
                with multiprocessing.Pool(processes=processes) as pool:
                    pool.map(time_resolved_rsa, [(args, n, True, searchlight_clusters) for n in range(1, experiment.subjects+1)])
                    pool.close()
                    pool.join()

        ### whole-trial encoding / decoding
        elif args.analysis in ['encoding', 'decoding']:
            print(args.evaluation_method, args.word_vectors)
            ### Checks
            if not args.evaluation_method and not args.word_vectors:
                raise RuntimeError('You need to specify both evaluation method '\
                                   'and the word vector model for encoding/decoding')
            ### Loading word vectors
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
                for n_ceiling in range(1, experiment.subjects+1):
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
                    comp_vectors = load_vectors_two(args, experiment, experiment.subjects)

            for n in range(1, experiment.subjects+1):
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

        elif 'classification' in args.analysis:

            ### Searchlight classification
            if 'searchlight' in args.analysis:
                for n in range(1, experiment.subjects+1):

                    ### Preparing clusters
                    experiment = ExperimentInfo(args, subject=n)
                    eeg = LoadEEG(args, experiment, n) 

                    tmin = 0.
                    tmax = .8
                    relevant_times = list(range(int(tmin*10000), int(tmax*10000), searchlight_clusters.time_radius))

                    electrode_indices = [searchlight_clusters.neighbors[center] for center in range(128)]
                    #clusters = [(e_s, t_s) for e_s in electrode_indices for t_s in relevant_times]
                    #for cluster in clusters:
                    out_times = list()
                    clusters = list()
                    for e_s in electrode_indices:
                        for t in relevant_times:
                            start_time = min([t_i for t_i, timepoint in enumerate(eeg.times) if timepoint*10000>t])
                            end_time = max([t_i for t_i, timepoint in enumerate(eeg.times) if timepoint*10000<=t+searchlight_clusters.time_radius])+1
                            out_times.append(start_time)

                            clusters.append((e_s, start_time, end_time))
                    out_times = sorted(list(set(out_times)))

                    print('Searchlight subject n. {}'.format(n))
                    if args.debugging:
                        results = list()
                        for cluster in clusters:
                            results.append(run_searchlight_classification([ 
                                                args, experiment, eeg, cluster]))
                    else:
                        with multiprocessing.Pool(processes=processes) as p:
                            results = p.map(run_searchlight_classification, 
                            [[args, experiment, eeg, cluster] for cluster in clusters])
                            p.terminate()
                            p.join()

                    results_array = join_searchlight_results(results, out_times)
                    write_plot_searchlight(args, n, [(t/10000)+(searchlight_clusters.time_radius/20000) for t in relevant_times], results_array)

            ### Time-resolved classification
            else:

                ### Single subject, for debugging
                if args.debugging:

                    for n in range(1, experiment.subjects+1):
                        print('\nSubject {}'.format(n))
                        run_time_resolved_classification([args, n])
                
                ### All subjects with a separate process, much quicker
                else:

                    with multiprocessing.Pool(processes=processes) as pool:
                        sub_scores = pool.map(run_time_resolved_classification, [(args, 
                                                n,
                                                ) for n in range(1, experiment.subjects+1)])
                        pool.close()
                        pool.join()
