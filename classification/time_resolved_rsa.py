import numpy
import os
import scipy

from scipy import stats
from tqdm import tqdm

from general_utils import evaluate_pairwise, split_train_test
from io_utils import ExperimentInfo, LoadEEG, prepare_folder
from word_vector_enc_decoding.read_word_vectors import WordVectors, load_vectors_two
from searchlight.searchlight_utils import SearchlightClusters

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
        
    comp_vectors = load_vectors_two(args, experiment, n)
    eeg = {experiment.trigger_to_info[k][0] : v for k, v in eeg.items()}

    ### Words
    stimuli = list(eeg.keys())
    if args.experiment_id == 'two' and args.word_vectors == 'ceiling':
        stimuli = [s for s in stimuli if s in comp_vectors.keys()]
    '''
    if args.word_vectors in ['coarse_category', 'famous_familiar', 'fine_category']:
        if args.word_vectors == 'coarse_category':
            categories = {v[0] : v[1] for v in experiment.trigger_to_info.values()}
        elif args.word_vectors == 'famous_familiar':
            if args.experiment_id == 'one':
                raise RuntimeError('There is no famous_familiar distinction for this experiment!')
            categories = {v[0] : v[2] for v in experiment.trigger_to_info.values()}
        elif args.word_vectors == 'fine_category':
            if args.experiment_id == 'two':
                raise RuntimeError('There is no famous_familiar distinction for this experiment!')
            categories = {v[0] : v[2] for v in experiment.trigger_to_info.values()}
        model_sims = [0. if categories[k_one]==categories[k_two] else 1. for k_one in stimuli for k_two in stimuli if k_one!=k_two]
    elif args.word_vectors == 'mixed_category':
        model_sims = list()
        for k_one in stimuli:
            for k_two in stimuli:
                if k_one != k_two:
                    if categories[k_one][0] == categories[k_two][0]:
                        if categories[k_one][1] == categories[k_two][1]:
                            model_sims.append(1. - 1.)
                        else:
                            model_sims.append(1. - 0.5)
                    else:
                        model_sims.append(1. - 0.)
    else:
    '''
    for s in stimuli:
        assert s in comp_vectors.keys()
    ### splitting into batches of jump=? to control word length
    if args.corrected:
        jump = 8
        stimuli_batches = [sorted(stimuli)[i:i+jump] for i in range(0, len(stimuli), int(jump/2)) if i<=len(stimuli)-jump]
        assert len(set([n for b in stimuli_batches for n in b])) == len(stimuli)
    else:
        stimuli_batches = [stimuli]
    model_sims = list()
    for batch in stimuli_batches:
        if type(comp_vectors[stimuli[0]]) in [int, float, numpy.float64]:
            batch_sims = [abs(comp_vectors[tst]-comp_vectors[tr]) for tr in batch for tst in batch if tr!=tst]
        else:
            batch_sims = [1. - scipy.stats.pearsonr(comp_vectors[k_one], comp_vectors[k_two])[0] for k_one in batch for k_two in batch if k_one!=k_two]
        model_sims.append(batch_sims)

    if not searchlight:
        sub_scores = list()
        for t in tqdm(range(len(all_eeg.times))):
            current_eeg = {k : v[:, t] for k, v in eeg.items()}
            scores = list()
            if args.analysis == 'time_resolved_rsa':
                batch_corr = list()
                for batch, model_batch in zip(stimuli_batches, model_sims):
                    eeg_sims = [1. - scipy.stats.pearsonr(current_eeg[k_one], current_eeg[k_two])[0] for k_one in batch for k_two in batch if k_one!=k_two]
                    corr = scipy.stats.pearsonr(model_batch, eeg_sims)[0]
                    scores.append(corr)
            if args.analysis == 'time_resolved_rsa_encoding':
                for split in experiment.test_splits:
                    train_true, test_true, train_samples, test_samples = split_train_test(args, split, current_eeg, experiment, comp_vectors)
                    if args.word_vectors in ['coarse_category', 'famous_familiar']:
                        if test_samples[0] == test_samples[1]:
                            continue
                    score = evaluate_pairwise(args, train_true, test_true, train_samples, test_samples)
                    ### subtracting .5 because it's random baseline
                    scores.append(score-.5)
                ### calling it 'corr', but not really a corr...
            corr = numpy.average(scores)
            sub_scores.append(corr)
        out_path = prepare_folder(args)
        file_path = os.path.join(out_path, 'sub_{:02}_{}.txt'.format(n, args.word_vectors))
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
        #step = 8
        #step = 26
        #relevant_times = [t_i for t_i, t in enumerate(all_eeg.times) if t_i+(step*2)<len(all_eeg.times)][::step]
        #explicit_times = [all_eeg.times[t] for t in relevant_times]
        #explicit_times = [all_eeg.times[t+int(step/2)] for t in relevant_times]
        #relevant_times = [t_i for t_i, t in enumerate(all_eeg.times) if t_i+searchlight_clusters.time_radius<len(all_eeg.times)][::searchlight_clusters.time_radius]
        tmin = -.1
        tmax = 1.2
        relevant_times = list(range(int0(tmin*10000), int0(tmax*1000), searchlight_clusters.time_radius))
        #relevant_times = [t_i for t_i, t in enumerate(all_eeg.times) if t+(searchlight_clusters.time_radius)/1000<max(all_eeg.times)]
        #explicit_times = [all_eeg.times[t+int(searchlight_clusters.time_radius/2)] for t in relevant_times]

        electrode_indices = [searchlight_clusters.neighbors[center] for center in range(128)]
        #clusters = [(e_s, t_s) for e_s in electrode_indices for t_s in relevant_times]
        #for cluster in clusters:
        for e_s in electrode_indices:
            #places = list(cluster[0])
            for t in relevant_times:
                start_time = min([t_i for t_i, t in enumerate(all_eeg.times) if t>(searchlight_clusters.time_radius/10000)])
                end_time = max([t_i for t_i, t in enumerate(all_eeg.times) if t<=(searchlight_clusters.time_radius)/10000])+1
                print([start_time, end_time])
                #start_time = cluster[1]
                #current_eeg = {k : v[places, start_time:start_time+(step*2)].flatten() for k, v in eeg.items()}
                #current_eeg = {k : v[places, start_time:start_time+searchlight_clusters.time_radius].flatten() for k, v in eeg.items()}
                current_eeg = {k : v[places, start_time:end_time].flatten() for k, v in eeg.items()}
                eeg_sims = [1. - scipy.stats.pearsonr(current_eeg[k_one], current_eeg[k_two])[0] for k_one in stimuli for k_two in stimuli if k_one!=k_two]
                corr = scipy.stats.pearsonr(model_sims, eeg_sims)[0]
                results_dict[(places[0], start_time)] = corr

        results_array = list()
        for e in range(128):
            e_row = list()
            for t in relevant_times:
                e_row.append(results_dict[(e, t)])
            results_array.append(e_row)

        results_array = numpy.array(results_array)

        output_folder = prepare_folder(args)

        out_file = os.path.join(
                                output_folder, 
                  '{}_sub-{:02}.rsa'.format(args.word_vectors, n)
                  )
        if searchlight:
            out_file = out_file.replace('.rsa', '_spatial_{}_temporal_{}.rsa'.format(args.searchlight_spatial_radius, args.searchlight_temporal_radius))
        with open(out_file, 'w') as o:
            for t in explicit_times:
                o.write('{}\t'.format(t))
            o.write('\n')
            for e in results_array:
                for t in e:
                    o.write('{}\t'.format(t))
                o.write('\n')
