import pickle
import os
import numpy
import itertools
import random
import scipy
import sklearn
import collections

from scipy import stats
from skbold.preproc import ConfoundRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

from general_utils import split_train_test, prepare_folder 
from io_utils import LoadEEG, ExperimentInfo


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
    if args.input_target_model == 'coarse_category':
        cat_index = 1
    if args.input_target_model == 'fine_category':
        cat_index = 2
    if args.input_target_model == 'famous_familiar':
        cat_index = 2

    train_true, test_true, train_samples, test_samples, train_lengths, test_lengths = split_train_test(args, split, eeg, experiment, [])

    if args.analysis == 'searchlight':
        number_iterations = [0]
    else:
        sample_shape = train_samples[0].shape
        assert len(sample_shape) == 2
        number_iterations = range(train_samples[0].shape[-1])

    iteration_scores = list()

    for t_i, t in enumerate(number_iterations):

        if args.analysis == 'searchlight':
            t_train = train_samples.copy()
            t_test = test_samples.copy()
        else:
            t_train = [erp[:, t] for erp in train_samples]
            t_test = [erp[:, t] for erp in test_samples]
        if args.corrected:
            if t_i == 0:
                print('\nnow correcting for word length!\n')
            cfr = ConfoundRegressor(confound=train_lengths, X=numpy.array(t_train.copy()))
            cfr.fit(numpy.array(t_train))
            t_train = cfr.transform(numpy.array(t_train))

        ### Differentiating between binary and multiclass classifier
        if args.mapping_model == 'support_vector':
            classifier = SVC()
        #classifier = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0, 100., 1000., 10000.))
        elif args.mapping_model == 'ridge':
            classifier = RidgeClassifier()
            #classifier = RidgeClassifierCV()

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

    places = list(cluster[0])
    start_time = cluster[1]
    end_time = cluster[2]
    s_data = {k : e[places, start_time:end_time].flatten() for k, e in eeg.data_dict.items()}
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
    out_path = prepare_folder(args)
    ###Writing to file
    
    if args.corrected:
        file_path = os.path.join(out_path, 'sub_{:02}_accuracy_corrected_scores.txt'.format(n))
    else:
        file_path = os.path.join(out_path, 'sub_{:02}_accuracy_uncorrected_scores.txt'.format(n))
    #file_path = os.path.join(out_path, 'sub_{:02}_{}_scores.txt'.format(n+1, score_type))
    with open(os.path.join(file_path), 'w') as o:
        for t in times:
            o.write('{}\t'.format(t))
        o.write('\n')
        for d in final_score:
            o.write('{}\t'.format(d))
