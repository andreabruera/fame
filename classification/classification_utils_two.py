import pickle
import os
import numpy
import itertools
import random
import scipy
import sklearn
import collections

from scipy import stats
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

from plot_scripts.plot_utils import fdr, plot_time_resolved_classification
from io_utils import prepare_folder, LoadEEG, ExperimentInfo

### Actual classification

def classify(arg):

    args = arg[0]
    experiment = arg[1]
    eeg = arg[2]
    split = arg[3]
    iteration = arg[4]

    test_samples = list()
    test_true = list()

    ### Selecting the relevant index for the trigger_to_info dictionary
    if 'coarse' in args.analysis:
        cat_index = 1
    elif 'fine' in args.analysis or 'famous' in args.analysis:
        cat_index = 2

    for trig in split:
        erps = eeg[trig]
        if not isinstance(erps, list):
            erps = [erps]
        test_samples.extend(erps)
        test_true.extend([experiment.trigger_to_info[trig][cat_index] for i in range(len(erps))])

    train_samples = list()
    train_true = list()
    for k, erps in eeg.items():
        if not isinstance(erps, list):
            erps = [erps]
        if k not in split:
            train_samples.extend(erps)
            train_true.extend([experiment.trigger_to_info[k][cat_index] for i in range(len(erps))])

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
    
    train_labels = len(list(set(train_true)))

    iteration_scores = list()

    if 'searchlight' not in args.analysis:
        sample_shape = train_samples[0].shape
        assert len(sample_shape) == 2
        number_iterations = range(train_samples[0].shape[-1])

    else:
        number_iterations = [0]

    for t in number_iterations:

        if 'searchlight' not in args.analysis:
            t_train = [erp[:, t] for erp in train_samples]
            t_test = [erp[:, t] for erp in test_samples]
        else:
            t_train = train_samples.copy()
            t_test = test_samples.copy()

        ### Differentiating between binary and multiclass classifier
        #classifier = SVC()
        classifier = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0, 100., 1000., 10000.))

        ### Fitting the model on the training data
        classifier.fit(t_train, train_true)

        ### Computing the accuracy
        accuracy = classifier.score(t_test, test_true)
        iteration_scores.append(accuracy)

    return iteration_scores

### Functions for searchlight classification

def run_searchlight_classification(all_args): 

    args = all_args[0]
    experiment = all_args[1]
    eeg = all_args[2]
    cluster = all_args[3]
    step = all_args[4]

    places = list(cluster[0])
    start_time = cluster[1]
    s_data = {k : e[places, start_time:start_time+(step*2)].flatten() for k, e in eeg.data_dict.items()}
    ### Creating the input for the classification function
    classification_inputs = [[args, experiment, s_data, split, iteration] for iteration, split in enumerate(experiment.test_splits)]

    sub_scores = list(map(classify, classification_inputs))
    final_score = numpy.average(sub_scores)

    return [(places[0], start_time), final_score]

### Functions for time-resolved classification

#def run_time_resolved_classification(args, test_splits, frequencies):
def run_time_resolved_classification(all_args):
    args = all_args[0]
    n = all_args[1]

    ### Loading the experiment
    experiment = ExperimentInfo(args, subject=n)
    ### Loading the EEG data
    all_eeg = LoadEEG(args, experiment, n)
    eeg = all_eeg.data_dict

    ### Creating the input for the classification function
    classification_inputs = [[args, experiment, eeg, split, iteration] for iteration, split in enumerate(experiment.test_splits)]

    ### Testing
    sub_scores = list(map(classify, tqdm(classification_inputs)))

    ### Averaging
    final_score = numpy.average(sub_scores, axis=0)
    print(numpy.average(final_score))
    assert len(all_eeg.times) == len(final_score)
    write_time_resolved_classification(n, args, final_score, all_eeg.times, all_eeg.frequencies)

    return final_score

def write_time_resolved_classification(n, args, final_score, times, frequencies):
    score_type = 'accuracy'
    out_path = prepare_folder(args)
    ###Writing to file
    
    if args.corrected:
        file_path = os.path.join(out_path, 'sub_{:02}_{}_corrected_scores.txt'.format(n, score_type))
    else:
        file_path = os.path.join(out_path, 'sub_{:02}_{}_uncorrected_scores.txt'.format(n, score_type))
    #file_path = os.path.join(out_path, 'sub_{:02}_{}_scores.txt'.format(n+1, score_type))
    with open(os.path.join(file_path), 'w') as o:
        for t in times:
            o.write('{}\t'.format(t))
        o.write('\n')
        for d in final_score:
            o.write('{}\t'.format(d))
