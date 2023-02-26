import numpy
import os
import random
import scipy

from scipy import stats
from skbold.preproc import ConfoundRegressor
from tqdm import tqdm

from general_utils import evaluate_pairwise, rsa_evaluation_round, prepare_folder
from io_utils import ExperimentInfo, LoadEEG
from read_word_vectors import load_vectors
from searchlight import SearchlightClusters

def time_resolved_rsa(all_args):

    args = all_args[0]
    n = all_args[1]
    searchlight = all_args[2]
    searchlight_clusters = all_args[3]

    ### Loading the experiment
    experiment = ExperimentInfo(args, subject=n)
    ### Loading the EEG data
    all_eeg = LoadEEG(args, experiment, n)
    eeg = all_eeg.data_dict
        
    comp_vectors = load_vectors(args, experiment, n)
    eeg = {experiment.trigger_to_info[k][0] : v for k, v in eeg.items()}

    ### Words
    stimuli = list(eeg.keys())
    if args.experiment_id == 'two' and args.input_target_model == 'ceiling':
        stimuli = [s for s in stimuli if s in comp_vectors.keys()]
    for s in stimuli:
        assert s in comp_vectors.keys()
    ### splitting into batches of jump=? to control word length
    stimuli_batches = list()
    for test_split in experiment.test_splits:
        stimuli_batches.append([experiment.trigger_to_info[k][0] for k in test_split])
    model_sims = list()
    for batch in stimuli_batches:
        if type(comp_vectors[stimuli[0]]) in [int, float, numpy.float64]:
            batch_sims = [abs(comp_vectors[tst]-comp_vectors[tr]) for tr in batch for tst in eeg.keys() if tst not in batch]
        else:
            batch_sims = [1. - scipy.stats.pearsonr(comp_vectors[k_one], comp_vectors[k_two])[0] for k_one in batch for k_two in eeg.keys() if k_two not in batch]
        model_sims.append(batch_sims)

    if not searchlight:
        sub_scores = list()
        for t in tqdm(range(len(all_eeg.times))):
            current_eeg = {k : v[:, t] for k, v in eeg.items()}

            corr = rsa_evaluation_round(args, experiment, current_eeg, stimuli_batches, model_sims, comp_vectors)
            sub_scores.append(corr)
        out_path = prepare_folder(args)
        file_path = os.path.join(out_path, 'sub_{:02}_{}.txt'.format(n, args.input_target_model))
        correction = 'corrected' if args.corrected else 'uncorrected'
        file_path = file_path.replace('.txt', '_{}_scores.txt'.format(correction))
        with open(os.path.join(file_path), 'w') as o:
            for t in all_eeg.times:
                o.write('{}\t'.format(t))
            o.write('\n')
            for d in sub_scores:
                o.write('{}\t'.format(d))
    else:
        results_dict = dict()
        tmin = -.1
        tmax = 1.2
        relevant_times = list(range(int(tmin*10000), int(tmax*10000), searchlight_clusters.time_radius))
        out_times = list()

        electrode_indices = [searchlight_clusters.neighbors[center] for center in range(128)]
        for places in electrode_indices:
            for time in relevant_times:
                start_time = min([t_i for t_i, t in enumerate(all_eeg.times) if t>(time/10000)])
                out_times.append(start_time)
                end_time = max([t_i for t_i, t in enumerate(all_eeg.times) if t<=(time+searchlight_clusters.time_radius)/10000])+1
                #print([start_time, end_time])
                current_eeg = {k : v[places, start_time:end_time].flatten() for k, v in eeg.items()}
                corr = rsa_evaluation_round(args, experiment, current_eeg, stimuli_batches, model_sims)
                results_dict[(places[0], start_time)] = corr
        out_times = sorted(set(out_times))

        results_array = list()
        for e in range(128):
            e_row = list()
            for time in out_times:
                e_row.append(results_dict[(e, time)])
            results_array.append(e_row)

        results_array = numpy.array(results_array)

        output_folder = prepare_folder(args)

        out_file = os.path.join(
                                output_folder, 
                  '{}_sub-{:02}.rsa'.format(args.input_target_model, n)
                  )
        if searchlight:
            out_file = out_file.replace(
                    '.rsa', 
                    '_spatial_{}_temporal_{}.rsa'.format(
                           args.searchlight_spatial_radius, 
                           args.searchlight_temporal_radius
                           )
                    )
        with open(out_file, 'w') as o:
            for t in out_times:
                t = all_eeg.times[t]+(searchlight_clusters.time_radius/20000)
                o.write('{}\t'.format(t))
            o.write('\n')
            for e in results_array:
                for t in e:
                    o.write('{}\t'.format(t))
                o.write('\n')
