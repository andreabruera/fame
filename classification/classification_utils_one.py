import pickle
import os
import numpy
import itertools
import random
import skbold
import sklearn
import collections
import scipy

from scipy import stats
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

from skbold.preproc import ConfoundRegressor

from plot_scripts.plot_utils import fdr, plot_time_resolved_classification
from io_utils import prepare_folder, LoadEEG


def run_classification(all_arguments):

    correlation_mapper = {'attore' : 14,
                          'musicista' : 9,
                          'scrittore' : 13,
                          'politico' : 13,
                          'persona' : 12, 
                          'luogo' : 9,
                          'città' : 6, 
                          'stato' : 7,
                          "corso d'acqua" : 12, 
                          'monumento' : 11
                          }

    n = all_arguments[0]
    args = all_arguments[1]
    experiment = all_arguments[2]
    if 'searchlight' in args.analysis:
        cluster = all_arguments[3]

    if args.semantic_category != 'people_and_places':
        coarse = 'persona' if args.semantic_category == 'people' else 'luogo'

    data_paths = experiment.eeg_paths
    out_path = prepare_folder(args)

    if 'whole_trial' in args.analysis:

        feature_selection_method = 'no_reduction' if args.feature_reduction == 'no_reduction' \
                                   else '{}_{}'.format(args.feature_selection_method, args.feature_reduction)
        out_path = os.path.join(out_path, feature_selection_method)
        os.makedirs(out_path, exist_ok=True)

    data_path = data_paths[n]

    eeg_data = LoadEEG(args, data_path, n, experiment)
    eeg = eeg_data.data_dict
    times = eeg_data.times
    if args.data_kind == 'time_frequency':
        frequencies = eeg_data.frequencies
    #print((n, times.shape))
    trigger_to_info = experiment.trigger_to_info

    ### Restricting the data if training/testing on individuals only
    if args.entities == 'individuals_only':
        eeg = {k : v for k, v in eeg.items() if k<=100}
        trigger_to_info = {k : v for k, v in trigger_to_info.items() if k <= 100}

    '''
    ### Changing labels in the case of word length
    if args.entities == '':
        
        trigger_to_info = {k : [v[i] if i!=1 else len(v[0]) for i in range(3)] for k, v in trigger_to_info.items()}
        median = [k[0] for k in sorted([(k, v[1]) for k, v in trigger_to_info.items()], key=lambda item : item[1])]
        median_mapper = {k : 0 if k_i<int(len(median)/2) else 1 for k_i, k in enumerate(trigger_to_info.keys())}

        trigger_to_info = {k : [v[i] if i!=1 else median_mapper[k] for i in range(3)] for k, v in trigger_to_info.items()}
    '''

    ### Restricting the data if training/testing on fine categories
    ### within each coarse category
    if args.semantic_category != 'people_and_places':
        trigger_to_info = {k : v for k, v in trigger_to_info.items() if v[1] == coarse}
        eeg = {k : eeg[k] for k, v in trigger_to_info.items()}

    ### Checking that everything went fine
    if args.semantic_category in ['people', 'places']:
        if args.entities != 'individuals_only':
            assert len(list(eeg.keys())) == 20
        else:
            assert len(list(eeg.keys())) ==16 
    else:
        if args.entities != 'individuals_only':
            assert len(list(eeg.keys())) == 40
        else:
            assert len(list(eeg.keys())) ==32 

    if 'searchlight' in args.analysis:
        eeg = {k : [v[cluster, :] for v in vec] for k, vec in eeg.items()}

    ### Splitting
    test_splits = split(args, eeg, trigger_to_info)

    ### Computing correlation
    split_corrs = list()
    cat_index = 1 if 'coarse' in args.analysis else 2
    for s in test_splits:
        trigs = s[0]
        labels = list()
        lengths = list()
        for t in trigs:
            labels.append(trigger_to_info[t][cat_index])
            lengths.append(len(trigger_to_info[t][0]))
        #labels = sklearn.preprocessing.LabelEncoder().fit_transform(labels)
        labels = [correlation_mapper[l] for l in labels]
        '''
        try:
            corr = scipy.stats.linregress(lengths, labels)[2:4]
        except FloatingPointError:
            corr = 0.0
        '''
        corr = list(scipy.stats.spearmanr(lengths, labels))
        #corr = list(scipy.stats.pearsonr(lengths, labels))
        split_corrs.append(corr)

    #split_corrs = sorted(enumerate(split_corrs), key=lambda item : abs(item[1][0]))
    split_corrs = sorted(enumerate(split_corrs), key=lambda item : abs(item[1][0]))

    corrected_n_samples = len([t for t in split_corrs if t[1][0]==0.0])
    if args.corrected:
        if corrected_n_samples >= 50:
            test_splits = [test_splits[t[0]] for t in split_corrs if t[1][0]==0.0]

        else:
            test_splits = [test_splits[t[0]] for t in split_corrs[:50]]
    else:
        n_samples = max(50, corrected_n_samples)
        test_splits = test_splits[:n_samples]
    print(split_corrs[49])

    ### Testing
    sub_scores = list(map(classify, tqdm(test_splits)))

    if args.data_kind == 'erp' or 'whole_trial' in args.analysis:

        ### Averaging
        final_score = numpy.average(sub_scores, axis=0)
        if 'whole_trial' not in args.analysis:
            assert len(times) == len(final_score)

        if 'searchlight' in args.analysis:
            score_type = 'cluster_{}_accuracy'.format(cluster[0])
        else:
            score_type = 'accuracy'

        ###Writing to file
        
        if args.corrected:
            file_path = os.path.join(out_path, 'sub_{:02}_{}_corrected_scores.txt'.format(n+1, score_type))
        else:
            file_path = os.path.join(out_path, 'sub_{:02}_{}_uncorrected_scores.txt'.format(n+1, score_type))
        with open(os.path.join(file_path), 'w') as o:
            if 'whole_trial' not in args.analysis:
                for t in times:
                    o.write('{}\t'.format(t))
            else:
                o.write('Whole-trial accuracy\n')
            o.write('\n')
            for d in final_score:
                o.write('{}\t'.format(d))

    elif args.data_kind == 'time_frequency':

        for f_i, f in enumerate(frequencies):
            ### Averaging
            final_score = numpy.average([score[f_i] for score in sub_scores], axis=0)
            if 'whole_trial' not in args.analysis:
                assert len(times) == len(final_score)
            score_type = 'accuracy'

            ###Writing to file
            
            file_path = os.path.join(out_path, '{}_hz_sub_{:02}_{}_scores.txt'.format(f, n+1, score_type))
            with open(os.path.join(file_path), 'w') as o:
                for t in times:
                    o.write('{}\t'.format(t))
                o.write('\n')
                for d in final_score:
                    o.write('{}\t'.format(d))

def split(args, eeg, trigger_to_info):

    ### Preparing the variables and adjusting the data
    if args.entities == 'individuals_only':

        if args.semantic_category != 'people_and_places':

            cat_index = 2
            cat_length = 4
            combinations = list(itertools.permutations(list(range(4)), r=4))

        else:

            if 'classification_coarse' in args.analysis:
                cat_index = 1
                cat_length = 16
                combinations_one_cat = list(itertools.combinations(list(range(16)), 2))
                combinations = list(itertools.product(combinations_one_cat, repeat=2))

            elif 'classification_fine' in args.analysis:
                cat_index = 2
                cat_length = 4
                combinations = list(itertools.product(list(range(4)), repeat=8))

    elif args.entities == 'individuals_and_categories':

        if args.semantic_category != 'people_and_places':

            cat_index = 2
            cat_length = 5
            combinations = list(itertools.permutations(list(range(5)), r=4))

        else:

            if 'classification_coarse' in args.analysis:

                cat_index = 1
                cat_length = 20
                combinations_one_cat = list(itertools.combinations(list(range(20)), 2))
                combinations = list(itertools.product(combinations_one_cat, repeat=2))

            elif 'classification_fine' in args.analysis:

                cat_index = 2
                cat_length = 5
                combinations = list(itertools.product(list(range(4)), repeat=10))

    elif args.entities == 'all_to_individuals':

        if args.semantic_category != 'people_and_places':

            cat_index = 2
            cat_length = 4
            combinations = list(itertools.permutations(list(range(4)), r=4))

        else:

            if 'classification_coarse' in args.analysis:

                cat_index = 1
                cat_length = 16
                combinations_one_cat = list(itertools.combinations(list(range(16)), 2))
                combinations = list(itertools.product(combinations_one_cat, repeat=2))

            elif 'classification_fine' in args.analysis:

                cat_index = 2
                cat_length = 4
                combinations = list(itertools.product(list(range(4)), repeat=8))
    
    ### TO BE FIXED
    elif args.entities == 'individuals_to_categories' or args.entities == 'all_to_categories':

        if args.semantic_category != 'people_and_places':

            cat_index = 1
            cat_length = 4
            combinations = list(list(list(range(4))))

        else:
            cat_index = 1
            cat_length = 4
            combinations_one_cat = list(itertools.combinations(list(range(cat_length)), 2))
            combinations = list(itertools.product(combinations_one_cat, repeat=2))

    ### Getting the list of categories
    cats = set([v[cat_index] for k, v in trigger_to_info.items()])
    fine_to_coarse = {v[2] : v[1] for k, v in trigger_to_info.items()}
    
    if args.entities == 'individuals_to_categories' or args.entities == 'all_to_categories':
        cat_to_trigger = {k : [trigger_id for trigger_id, cat in trigger_to_info.items() \
                                                      if cat[cat_index]==k and trigger_id>100] for k in cats}
    elif args.entities == 'all_to_individuals':
        cat_to_trigger = {k : [trigger_id for trigger_id, cat in trigger_to_info.items() \
                                                      if cat[cat_index]==k and trigger_id<=100] for k in cats}
    else:
        cat_to_trigger = {k : [trigger_id for trigger_id, cat in trigger_to_info.items() \
                                                      if cat[cat_index]==k] for k in cats}

    for k, v in cat_to_trigger.items():
        assert len(v) == cat_length

    ### k-fold cross-validation with 80/20 splits
    if args.data_kind == 'erp':
        folds = 8192
    elif args.data_kind == 'time_frequency':
        folds = 1024
        #folds = 12
    test_permutations = list(random.sample(combinations, k=len(combinations)))[:folds]

    if isinstance(test_permutations[0], int):
        test_permutations = [test_permutations]

    test_splits = list()
    for i_p, p in enumerate(test_permutations):

        triggers = list()

        if isinstance(p[0], tuple): ### coarse
            for kv_i, kv in enumerate(cat_to_trigger.items()):
                for ind in p[kv_i]:
                    triggers.append(kv[1][ind])

        else:  ### fine
            for kv_i, kv in enumerate(cat_to_trigger.items()):
                triggers.append(kv[1][p[kv_i]])

        test_splits.append([triggers, eeg, trigger_to_info, i_p, args])

    return test_splits

def classify(arg):

    split = arg[0]
    eeg = arg[1]
    trigger_to_info = arg[2]
    iteration = arg[3]
    args = arg[4]

    test_samples = list()
    test_true = list()
    test_confounds = list()

    ### Selecting the relevant index for the trigger_to_info dictionary
    if args.semantic_category != 'people_and_places':
        cat_index = 2

    else:
        if 'classification_coarse' in args.analysis:
            cat_index = 1
        elif 'classification_fine' in args.analysis:
            cat_index = 2

        #elif args.entities == 'individuals_to_categories':
        #    cat_index = 1

    ### Test set
    for trig in split:

        erps = eeg[trig]
        if not isinstance(erps, list):
            erps = [erps]
        test_samples.extend(erps)
        test_true.extend([trigger_to_info[trig][cat_index] for i in range(len(erps))])
        test_confounds.append(len(trigger_to_info[trig][0]))

    ### Train set
    if args.entities == 'individuals_to_categories':
        eeg = {k : v for k, v in eeg.items() if k <= 100}
    train_samples = list()
    train_true = list()
    train_confounds = list()
    for k, erps in eeg.items():
        if not isinstance(erps, list):
            erps = [erps]
        if k not in split:
            train_samples.extend(erps)
            train_true.extend([trigger_to_info[k][cat_index] for i in range(len(erps))])

            train_confounds.append(len(trigger_to_info[k][0]))

    #print(len(train_true))

    train_true = LabelEncoder().fit_transform(train_true)
    test_true = LabelEncoder().fit_transform(test_true)

    sample_shape = train_samples[0].shape

    iteration_scores = list()

    if 'whole_trial' not in args.analysis:
        if args.data_kind == 'erp':
            assert len(sample_shape) == 2
        elif args.data_kind == 'time_frequency':
            assert len(sample_shape) == 3

        ### ERP
        if len(sample_shape) == 2:

            for t in range(train_samples[0].shape[-1]):

                t_train = [erp[:, t] for erp in train_samples]
                t_test = [erp[:, t] for erp in test_samples]

                '''
                ### Controlling for length
                confound_controller = ConfoundRegressor(confound=numpy.array(train_confounds+test_confounds), \
                                                        X=numpy.array(t_train+t_test))
                t_train = confound_controller.fit_transform(numpy.array(t_train))
                t_test = confound_controller.transform(numpy.array(t_test))
                '''

                ### Differentiating between binary and multiclass classifier
                if 'classification_coarse' in args.analysis:
                    classifier = SVC()
                elif 'classification_fine' in args.analysis:
                    classifier = OneVsRestClassifier(SVC())

                svc_pipeline = Pipeline([('svc', classifier)])

                ### Fitting the model on the training data
                svc_pipeline.fit(t_train, train_true)

                ### Computing the accuracy
                accuracy = svc_pipeline.score(t_test, test_true)
                '''
                ### Area under the curve
                predicted = svc_pipeline.predict(t_test)
                auc = sklearn.metrics.roc_auc_score(test_true, predicted)

                iteration_scores.append([accuracy, auc])
                '''
                iteration_scores.append(accuracy)

        ### Time-frequency 
        if len(sample_shape) == 3:

            sample_frequencies = sample_shape[-2]
            sample_times = sample_shape[-1]

            for s_f in range(sample_frequencies):
                frequency_scores = list()
            
                for t in range(sample_times):

                    t_train = [erp[:, s_f, t] for erp in train_samples]
                    t_test = [erp[:, s_f, t] for erp in test_samples]

                    ### Differentiating between binary and multiclass classifier
                    if 'classification_coarse' in args.analysis:
                        classifier = SVC()
                    elif 'classification_fine' in args.analysis:
                        classifier = OneVsRestClassifier(SVC())

                    svc_pipeline = Pipeline([('svc', classifier)])

                    ### Fitting the model on the training data
                    svc_pipeline.fit(t_train, train_true)

                    ### Computing the accuracy
                    accuracy = svc_pipeline.score(t_test, test_true)
                    '''
                    ### Area under the curve
                    predicted = svc_pipeline.predict(t_test)
                    auc = sklearn.metrics.roc_auc_score(test_true, predicted)

                    iteration_scores.append([accuracy, auc])
                    '''
                    frequency_scores.append(accuracy)

                iteration_scores.append(numpy.array(frequency_scores))

    else:
        assert len(sample_shape) == 1
        assert sample_shape[0] == int(args.feature_reduction)
        ### ERP

        t_train = [erp.flatten() for erp in train_samples]
        t_test = [erp.flatten() for erp in test_samples]

        ### Differentiating between binary and multiclass classifier
        if 'classification_coarse' in args.analysis:
            classifier = SVC()
        elif 'classification_fine' in args.analysis:
            classifier = OneVsRestClassifier(SVC())

        svc_pipeline = Pipeline([('svc', classifier)])

        ### Fitting the model on the training data
        svc_pipeline.fit(t_train, train_true)

        ### Computing the accuracy
        accuracy = svc_pipeline.score(t_test, test_true)
        '''
        ### Area under the curve
        predicted = svc_pipeline.predict(t_test)
        auc = sklearn.metrics.roc_auc_score(test_true, predicted)

        iteration_scores.append([accuracy, auc])
        '''
        iteration_scores.append(accuracy)

    iteration_scores = numpy.array(iteration_scores)
    #print(iteration_scores.shape)

    return iteration_scores
