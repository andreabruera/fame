import numpy
import os
import scipy

from scipy import stats

from io_utils import ExperimentInfo, LoadEEG, prepare_folder
from word_vector_enc_decoding.read_word_vectors import WordVectors, load_vectors_two

def time_resolved_rsa(all_args):
    args = all_args[0]
    n = all_args[1]

    ### Loading the experiment
    experiment = ExperimentInfo(args, subject=n)
    ### Loading the EEG data
    all_eeg = LoadEEG(args, experiment, n)
    eeg = all_eeg.data_dict
    if args.word_vectors == 'ceiling':
        ceiling = dict()
        for n_ceiling in range(33):
            eeg_data_ceiling = LoadEEG(args, experiment, n_ceiling+1)
            eeg_ceiling = eeg_data_ceiling.data_dict
            for k, v in eeg_ceiling.items():
                ### Adding and flattening
                if k not in ceiling.keys():
                    ceiling[k] = list()
                ceiling[k].append(v.flatten())
        comp_vectors = {k : numpy.average([vec for vec_i, vec in enumerate(v) if vec_i!=n-1], axis=0) for k, v in ceiling.items()}
    elif args.word_vectors == 'coarse_category':
        categories = {v[0] : v[1] for v in experiment.trigger_to_info.values()}
    elif args.word_vectors == 'famous_familiar':
        if args.experiment_id == 'one':
            raise RuntimeError('There is no famous_familiar distinction for this experiment!')
        categories = {v[0] : v[2] for v in experiment.trigger_to_info.values()}
    elif args.word_vectors == 'fine_category':
        if args.experiment_id == 'two':
            raise RuntimeError('There is no famous_familiar distinction for this experiment!')
        categories = {v[0] : v[2] for v in experiment.trigger_to_info.values()}
    elif args.word_vectors == 'mixed_category':
        if args.experiment_id == 'two':
            raise RuntimeError('There is no famous_familiar distinction for this experiment!')
        categories = {v[0] : (v[1], v[2]) for v in experiment.trigger_to_info.values()}
        
    else:
        comp_vectors = load_vectors_two(args, experiment, n)
    eeg = {experiment.trigger_to_info[k][0] : v for k, v in eeg.items()}

    ### Words
    stimuli = list(eeg.keys())
    if args.word_vectors in ['coarse_category', 'famous_familiar', 'fine_category']:
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
        for s in stimuli:
            assert s in comp_vectors.keys()
        model_sims = [1. - scipy.stats.pearsonr(comp_vectors[k_one], comp_vectors[k_two])[0] for k_one in stimuli for k_two in stimuli if k_one!=k_two]

    sub_scores = list()
    for t in range(len(all_eeg.times)):
        current_eeg = {k : v[:, t] for k, v in eeg.items()}
        eeg_sims = [1. - scipy.stats.pearsonr(current_eeg[k_one], current_eeg[k_two])[0] for k_one in stimuli for k_two in stimuli if k_one!=k_two]
        corr = scipy.stats.pearsonr(model_sims, eeg_sims)[0]
        sub_scores.append(corr)
    out_path = prepare_folder(args)
    file_path = os.path.join(out_path, 'sub_{:02}_{}.txt'.format(n, args.word_vectors))
    with open(os.path.join(file_path), 'w') as o:
        for t in all_eeg.times:
            o.write('{}\t'.format(t))
        o.write('\n')
        for d in sub_scores:
            o.write('{}\t'.format(d))
