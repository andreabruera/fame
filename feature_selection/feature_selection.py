import argparse
import itertools
import logging
import multiprocessing
import numpy
import os
import sys

#sys.path.append('/import/cogsci/andrea/github/fame')
from io_utils import ExperimentInfo, LoadEEG, prepare_folder
from searchlight.searchlight_utils import SearchlightClusters, run_searchlight
from feature_selection.scores import get_scores

from scipy import stats

def feature_ranking_folder(args, n):
    #out_path = os.path.join('feature_ranking', args.data_kind, \
    #                        'sub-{:02}'.format(n+1), 'people_and_places', \
    #                        args.entities, args.subsample)
    out_path = prepare_folder(args).replace('feature_selection', 'feature_scores')

    os.makedirs(out_path, exist_ok=True)

    return out_path

def compute_and_write(all_args):

    n = all_args[0]
    data_path = all_args[1]
    args = all_args[2]
    trigger_to_info = all_args[3]
    experiment = all_args[4]

    ### Filtering the words to be used
    if args.entities == 'individuals_only':
        selected_triggers = [k for k, v in trigger_to_info.items() if k <= 100]
    else:
        selected_triggers = list(trigger_to_info.keys())
    '''
    elif args.restrict_words == 'people':
        selected_triggers = [k for k, v in trigger_to_info.items() if v[1] == 'persona']
    elif args.restrict_words == 'places':
        selected_triggers = [k for k, v in trigger_to_info.items() if v[1] == 'luogo']
    '''

    ### Reading the EEG data
    eeg_data = LoadEEG(args, data_path, n, experiment)
    full_eeg = eeg_data.full_data
    ### Selecting relevant triggers
    full_eeg = {k : v for k, v in full_eeg.items() if k in selected_triggers}
    '''

    #logging.info('Now computing the scores for subject {}'.format(n+1))
    ### Computing the scores
    if args.data_kind == 'erp':
        hz_freqs = ['']
    else:
        hz_freqs = eeg_data.frequencies
    
    for hz_i, hz in enumerate(hz_freqs):

        if hz == '':
            eeg = full_eeg
        else:
            eeg = {k : v[:, :, hz_i, :] for k, v in full_eeg.items()}
    '''
    hz = ''
    eeg = full_eeg

    scores = get_scores(args, experiment, eeg)

    '''
    out_path = feature_ranking_folder(args, n)
    #out_path = os.path.join(out_path, '{}.scores'.format(feature))
                            
    ### Writing to file
    for score in scores:

        score_type = score[0]
        score_values = score[1]
        
        hz_marker = '' if hz=='' else '{}hz_'.format(hz)
        with open(os.path.join(out_path, '{}{}.scores'.format(\
                             hz_marker, score_type)), 'w') as o:
            o.write('Word 1\tWord 2\tFeatures ordered by decreasing score\n')
            for comb, values in score_values.items():
                words = (trigger_to_info[c][0] for c in comb)
                for w in words:
                    o.write('{}\t'.format(w))
                for dimension, value in values:
                    o.write('{},{}\t'.format(dimension, value))
                o.write('\n')
    '''
    return scores
