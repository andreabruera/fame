import argparse
import os
import collections
import numpy
import re
import logging
import itertools
import functools
import mne
import pickle
import scipy
import multiprocessing

from scipy import stats
from matplotlib import pyplot
from tqdm import tqdm

from io_utils import prepare_folder
from searchlight.searchlight_utils import SearchlightClusters

from io_utils import ExperimentInfo, LoadEEG

def feature_selection_group_searchlight(args, exp, hz=''):

    out_path = prepare_folder(args).replace('feature_selection', 'feature_scores')
    os.makedirs(out_path, exist_ok=True)

    electrode_index_to_code = SearchlightClusters().index_to_code
    mne_adj_matrix = SearchlightClusters().mne_adjacency_matrix
    sub_collector = list()
    for n_one in range(1, exp.subjects+1):

        all_subjects = list()

        for n in range(1, exp.subjects+1):
            if n != n_one:

                input_folder = prepare_folder(args).replace('feature_selection', 'rsa_searchlight')
                #.replace('group', 'rsa').replace('feature_selection_', '')
                with open(os.path.join(input_folder, '{}_{}sub-{:02}.rsa'.format(args.word_vectors, hz, n)), 'r') as i:
                    lines = [l.strip().split('\t') for l in i.readlines()]
                times = [float(w) for w in lines[0]]
                electrodes = numpy.array([[float(v) for v in l] for l in lines[1:]]).T
                all_subjects.append(electrodes)

        all_subjects = numpy.array(all_subjects)
        assert all_subjects.shape[0] == 32

        t_stats, _, \
        p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(all_subjects, \
                                                           tail=1, \
                                                           adjacency=mne_adj_matrix, \
                                                           threshold=dict(start=0, step=0.2), \
                                                           n_jobs=int(os.cpu_count()/2), \
                                                           #n_permutations=4096, \
                                                           n_permutations=1024, \
                                                           #n_permutations='all', \
                                                           )
        t_stats = t_stats.T
        assert t_stats.shape[0] == 128
        sub_collector.append(t_stats)
    sub_collector = numpy.array(sub_collector)
        
    return sub_collector
