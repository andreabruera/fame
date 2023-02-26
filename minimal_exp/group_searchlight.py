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

from general_utils import prepare_folder
from io_utils import ExperimentInfo, LoadEEG
from searchlight import SearchlightClusters

def group_searchlight(args):

    print(args.analysis)
    #excluded_subjects = [18, 28, 31]
    pyplot.rcParams['figure.constrained_layout.use'] = True

    plot_path = prepare_folder(args).replace('results', 'plots')
    #significance = .005
    #significance = 0.1
    clusters = SearchlightClusters(args)
    electrode_index_to_code = clusters.index_to_code
    mne_adj_matrix = clusters.mne_adjacency_matrix
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

        ### adding information about cluster_size
        input_file = input_file.replace('.rsa', 
                     '_spatial_{}_temporal_{}.rsa'.format(args.searchlight_spatial_radius, args.searchlight_temporal_radius))

        assert os.path.exists(input_file)
        with open(input_file, 'r') as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        ### reading times and electrodes
        times = numpy.array([w for w in lines[0]], dtype=numpy.float64)
        electrodes = numpy.array([[float(v) for v in l] for l in lines[1:]]).T
        assert electrodes.shape == (len(times), 128)
        ### reducing relevant time points
        ### following leonardelli & fairhall, range is 100-750ms
        relevant_indices = [t_i for t_i, t in enumerate(times) if (t>0.2 and t<0.6)]
        #relevant_indices = [t_i for t_i, t in enumerate(times) if t>0.]
        times = times[relevant_indices]
        electrodes = electrodes[relevant_indices, :]

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
                                                       #n_permutations=4000, \
                                                       #n_permutations=10000, 
                                                       #n_permutations='all', \
                                                       )
    #print(min(t_stats.flatten()))
    ### Plotting the results
    if 'classification' in args.analysis:
        logging.info('Minimum p-value for {}: {}'.format(args.semantic_category, min(p_values)))
    else:
        logging.info('Minimum p-value for {}: {}'.format(args.word_vectors, min(p_values)))

    significance = .05
    original_shape = t_stats.shape
    
    #log_p[log_p<=-numpy.log(0.05)] = 0.0

    for significance in [0.05, #0.1
                        ]:

        reshaped_p = p_values.copy()
        reshaped_p[reshaped_p>=significance] = 1.0
        reshaped_p = reshaped_p.reshape(original_shape).T

        ### relevant time indices
        significant_times = list()
        for t in range(reshaped_p.shape[-1]):
            if numpy.sum(reshaped_p[:, t]) < reshaped_p.shape[0]:
                significant_times.append(times[t])

        #significant_points = res[2].reshape(res[0].shape).T
        #significant_points = -numpy.log(significant_points)
        #significant_points[significant_points<=-numpy.log(0.05)] = 0.0

        #significant_indices = [i[0] for i in enumerate(numpy.nansum(log_p.T, axis=1)>0.) if i[1]==True]
        #significant_times = [times[i] for i in significant_indices if i!=0]
        #semi_sig_indices = [p for p in p_values if (p<=0.08 and p>0.05)]

        print(significant_times)
        #print('number of semi significant time points: {}'.format(len(semi_sig_indices)))

        #relevant_times
        tmin = times[0]
        if args.data_kind != 'time_frequency':
            #sfreq = 256/8
            if args.searchlight_temporal_radius == 'large':
                sfreq = 10
            elif args.searchlight_temporal_radius == 'medium':
                sfreq = 20 
            elif args.searchlight_temporal_radius == 'small':
                sfreq = 40

        elif args.data_kind == 'time_frequency':
            sfreq = 256 / 16
        info = mne.create_info(ch_names=[v for k, v in clusters.index_to_code.items()], \
                               sfreq=sfreq, \
                               ch_types='eeg')

        log_p = -numpy.log(p_values)

        #log_p[p_values>=significance] = .0

        log_p = log_p.reshape(original_shape).T
        #evoked = mne.EvokedArray(log_p, info=info, tmin=tmin)
        evoked = mne.EvokedArray(t_stats.T, info=info, tmin=tmin)

        #evoked = mne.EvokedArray(reshaped_p, info=info, tmin=tmin)

        montage = mne.channels.make_standard_montage('biosemi128')
        evoked.set_montage(montage)

        #if len(significant_times) >= 1:
        os.makedirs(plot_path, exist_ok=True)

        ### Writing to txt
        channels = evoked.ch_names
        assert isinstance(channels, list)
        #assert len(channels) == log_p.shape[0]
        assert len(channels) == reshaped_p.shape[0]
        assert len(times) == reshaped_p.shape[-1]
        if 'classification' in args.analysis:
            txt_path = os.path.join(plot_path, 'searchlight_classification_significant_points_{}.txt'.format(significance))
        else:
            txt_path = os.path.join(plot_path, '{}_searchlight_rsa_significant_points_{}.txt'.format(args.word_vectors, significance))

        txt_path = txt_path.replace('.txt', '_spatial_{}_temporal_{}.txt'.format(
                                                   args.searchlight_spatial_radius,
                                                   args.searchlight_temporal_radius,
                                                   ))
        with open(txt_path, 'w') as o:
            o.write('Time\tElectrode\tp-value\tt-value\n')
            for t_i in range(reshaped_p.shape[-1]):
                time = times[t_i]
                for c_i in range(reshaped_p.shape[0]):
                    channel = channels[c_i]
                    p = reshaped_p[c_i, t_i]
                    if p != 0.:
                        p_value = reshaped_p[c_i, t_i]
                        t_value = t_stats.T[c_i, t_i]
                        o.write('{}\t{}\t{}\t{}\n'.format(time, channel, p_value, t_value))
        #for i in range(2):

        #mode = 'all' if i==0 else 'significant'

        correction = 'corrected' if args.corrected else 'uncorrected'
        if 'classification' in args.analysis:
            title = '{} Classification Searchlight for: {}'.format(re.sub('^.+?classification', '', args.analysis), args.semantic_category).replace('_', ' ')
        else:
            title='RSA Searchlight for {} - {}'.format(args.word_vectors, args.semantic_category.replace('_', ' '))
        title = '{} - {}, p<={}'.format(title, correction, significance)

        #if mode == 'significant':
        evoked.plot_topomap(ch_type='eeg', 
                            time_unit='s', 
                            #times=significant_times,
                            #times=times-.025,
                            times=evoked.times,
                            ncols='auto',
                            nrows='auto', 
                            #outlines='skirt',
                            #vmin=-numpy.log(significance), 
                            vmin=-1.,
                            vmax=10.,
                            #vmin=0.000001,
                            scalings={'eeg':1.}, 
                            #cmap='YlGnBu', 
                            #cmap='bone_r',
                            cmap='Spectral_r',
                            #cmap='cividis',
                            mask=reshaped_p<=significance,
                            mask_params=dict(marker='o', markerfacecolor='black', markeredgecolor='black',
                                        linewidth=0, markersize=4),
                            #colorbar=False,
                            size = 3.,
                            title=title,
                            )
        #else:
            #evoked.plot_topomap(ch_type='eeg', time_unit='s', times=[i for i in evoked.times], units='-log(p)\nif\np<=.05', ncols=12, nrows='auto', vmin=0., scalings={'eeg':1.}, cmap='PuBu', title=title)

        f_name = os.path.join(plot_path, '{}_{}_{}_significant_points.jpg'.format(correction, args.semantic_category, significance))
        if 'classification' in args.analysis:
            f_name = f_name.replace('.jpg', 'classification_{}.jpg'.format(args.data_kind))
        else:
            f_name = f_name.replace('.jpg', '_rsa_{}.jpg'.format(args.word_vectors))
        f_name = f_name.replace('.jpg', '_spatial_{}_temporal_{}.jpg'.format(args.searchlight_spatial_radius, args.searchlight_temporal_radius))
        print(f_name)
        pyplot.savefig(f_name, dpi=600)
        pyplot.savefig(f_name.replace('jpg', 'svg'), dpi=600)
        pyplot.clf()

        #### Saving npy file
        #f_name = os.path.join(plot_path, '{}_significant_points.npy'.format(args.semantic_category))
        #if 'classification' in args.analysis:
        #    f_name = f_name.replace('.npy', 'classification.npy')
        #else:
        #    f_name = f_name.replace('.npy', 'rsa_{}.npy'.format(args.word_vectors))
        #numpy.save(f_name, reshaped_p)
