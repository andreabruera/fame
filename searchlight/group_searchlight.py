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

def group_searchlight(args):
    print(args.analysis)
    #excluded_subjects = [18, 28, 31]
    pyplot.rcParams['figure.constrained_layout.use'] = True

    plot_path = prepare_folder(args).replace('results', 'plots')
    #significance = .005
    significance = .05
    #significance = 0.1
    electrode_index_to_code = SearchlightClusters().index_to_code
    mne_adj_matrix = SearchlightClusters(max_distance=20).mne_adjacency_matrix
    all_subjects = list()

    for n in range(1, 34):
        #if 'classification' not in args.analysis:
        #    input_folder = prepare_folder(args)
        #else:
        #    input_folder = prepare_folder(args)
        input_folder = prepare_folder(args)
        if 'classification' in args.analysis:
            input_file = os.path.join(input_folder, 'sub-{:02}.rsa'.format(n))
        else:
            input_file = os.path.join(input_folder, '{}_sub-{:02}.rsa'.format(args.word_vectors, n))
        with open(input_file, 'r') as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        times = [float(w) for w in lines[0]]
        electrodes = numpy.array([[float(v) for v in l] for l in lines[1:]]).T
        all_subjects.append(electrodes)

    all_subjects = numpy.array(all_subjects)

    if 'rsa' not in args.analysis:
        if args.experiment_id == 'two':
            all_subjects = all_subjects-0.5
        elif args.experiment_id == 'one':
            if 'coarse' in args.analysis:
                all_subjects = all_subjects-0.5
            else:
                if args.semantic_category == 'all':
                    all_subjects = all_subjects-0.125
                else:
                    all_subjects = all_subjects-0.25


    t_stats, _, \
    p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(all_subjects, \
                                                       tail=1, \
                                                       adjacency=mne_adj_matrix, \
                                                       threshold=dict(start=0, step=0.2), \
                                                       n_jobs=os.cpu_count()-1, \
                                                       n_permutations=4000, \
                                                       #n_permutations=10000, 
                                                       #n_permutations='all', \
                                                       )

    ### Plotting the results
    if 'classification' in args.analysis:
        logging.info('Minimum p-value for {}: {}'.format(args.semantic_category, min(p_values)))
    else:
        logging.info('Minimum p-value for {}: {}'.format(args.word_vectors, min(p_values)))

    original_shape = t_stats.shape
    #reshaped_p = [val if val<=significance else 1. for val in p_values]
    reshaped_p = p_values.reshape(original_shape).T
    
    log_p = -numpy.log(p_values)
    #log_p[log_p<=-numpy.log(0.05)] = 0.0
    log_p[log_p<=-numpy.log(significance)] = 0.0

    log_p = log_p.reshape(original_shape).T

    #significant_points = res[2].reshape(res[0].shape).T
    #significant_points = -numpy.log(significant_points)
    #significant_points[significant_points<=-numpy.log(0.05)] = 0.0

    significant_indices = [i[0] for i in enumerate(numpy.nansum(log_p.T, axis=1)>0.) if i[1]==True]
    significant_times = [times[i] for i in significant_indices if i!=0]
    print(significant_times)

    #relevant_times
    tmin = times[0]
    if args.data_kind != 'time_frequency':
        #sfreq = 256/8
        sfreq = 256/26
    elif args.data_kind == 'time_frequency':
        sfreq = 256 / 16
    info = mne.create_info(ch_names=[v for k, v in SearchlightClusters().index_to_code.items()], \
                           #the step is 8 samples, so we divide the original one by 7
                           sfreq=sfreq, \
                           ch_types='eeg')

    evoked = mne.EvokedArray(log_p, info=info, tmin=tmin)
    #evoked = mne.EvokedArray(reshaped_p, info=info, tmin=tmin)

    montage = mne.channels.make_standard_montage('biosemi128')
    evoked.set_montage(montage)

    if len(significant_times) >= 1:
        os.makedirs(plot_path, exist_ok=True)

        ### Writing to txt
        channels = evoked.ch_names
        assert isinstance(channels, list)
        assert len(channels) == log_p.shape[0]
        assert len(times) == log_p.shape[-1]
        if 'classification' in args.analysis:
            txt_path = os.path.join(plot_path, 'searchlight_classification_significant_points.txt')
        else:
            txt_path = os.path.join(plot_path, '{}_searchlight_rsa_significant_points.txt'.format(args.word_vectors))
        with open(txt_path, 'w') as o:
            o.write('Time\tElectrode\tp-value\n')
            for t_i in range(log_p.shape[-1]):
                time = times[t_i]
                for c_i in range(log_p.shape[0]):
                    channel = channels[c_i]
                    p = log_p[c_i, t_i]
                    if p != 0.:
                        p_value = reshaped_p[c_i, t_i]
                        o.write('{}\t{}\t{}\n'.format(time, channel, p_value))
        #for i in range(2):

        #mode = 'all' if i==0 else 'significant'

        correction = 'corrected' if args.corrected else 'uncorrected'
        if 'classification' in args.analysis:
            title = '{} Classification Searchlight for: {}'.format(re.sub('^.+?classification', '', args.analysis), args.semantic_category).replace('_', ' ')
        else:
            title='RSA Searchlight for {} - {}'.format(args.word_vectors, args.semantic_category.replace('_', ' '))
        title = '{} - {}'.format(title, correction)

        if len(significant_times)  > 4:
            ncolumns = len(significant_times)
        else:
            ncolumns = 'auto'
        #if mode == 'significant':
        evoked.plot_topomap(ch_type='eeg', 
                            time_unit='s', 
                            times=significant_times,
                            ncols=ncolumns,
                            nrows='auto', 
                            outlines='skirt',
                            #vmin=-numpy.log(significance), 
                            #vmax=0.,
                            vmin=0.,
                            scalings={'eeg':1.}, 
                            cmap='YlGnBu', 
                            colorbar=False,
                            size = 3.,
                            title=title)
        #else:
            #evoked.plot_topomap(ch_type='eeg', time_unit='s', times=[i for i in evoked.times], units='-log(p)\nif\np<=.05', ncols=12, nrows='auto', vmin=0., scalings={'eeg':1.}, cmap='PuBu', title=title)

        f_name = os.path.join(plot_path, '{}_{}_significant_points.jpg'.format(correction, args.semantic_category))
        if 'classification' in args.analysis:
            f_name = f_name.replace('.jpg', 'classification_{}.jpg'.format(args.data_kind))
        else:
            f_name = f_name.replace('.jpg', '_rsa_{}.jpg'.format(args.word_vectors))
        print(f_name)
        pyplot.savefig(f_name, dpi=600)
        pyplot.clf()

        ### Saving npy file
        f_name = os.path.join(plot_path, '{}_significant_points.npy'.format(args.semantic_category))
        if 'classification' in args.analysis:
            f_name = f_name.replace('.npy', 'classification.npy')
        else:
            f_name = f_name.replace('.npy', 'rsa_{}.npy'.format(args.word_vectors))
        numpy.save(f_name, reshaped_p)
