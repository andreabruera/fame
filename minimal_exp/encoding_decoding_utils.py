import mne
import os
import sklearn
import itertools
import numpy
import random
import re
import scipy
import sklearn

from matplotlib import pyplot
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression, \
                                      mutual_info_regression
from tqdm import tqdm
from scipy import stats
from mne.decoding import CSP, SPoC

from general_utils import prepare_folder
from io_utils import LoadEEG

def write_enc_decoding(args, accuracies, word_by_word):

    out_path = prepare_folder(args)

    feature_selection_method = 'no_reduction' if args.feature_reduction == 'no_reduction' \
                               else '{}_{}'.format(args.feature_selection_method, args.feature_reduction)

    out_path = os.path.join(out_path, feature_selection_method)
    os.makedirs(out_path, exist_ok=True)
    file_name = '{}_{}_results.txt'.format(args.word_vectors, \
                 args.evaluation_method)
    ### Writing to file
    with open(os.path.join(out_path, file_name), 'w') as o:
        o.write('Accuracy\t')
        for w in word_by_word.keys():
            o.write('{}\t'.format(w))
        o.write('\n')
        for a_i, a in enumerate(accuracies):
            o.write('{}\t'.format(a))
            for w, accs in word_by_word.items():
                try:
                    acc = accs[a_i]
                    o.write('{}\t'.format(acc))
                except IndexError:
                    o.write('\t')
            o.write('\n')

def prepare_and_test(n, args, experiment, comp_vectors):

    cat = args.semantic_category

    numpy.seterr(all='raise')

    data_paths = experiment.eeg_paths

    data_path = data_paths[n]
    eeg_data = LoadEEG(args, experiment, n)
    ### Flattening
    eeg = {k : v.flatten() for k, v in eeg_data.data_dict.items()}
    times = eeg_data.times
    trigger_to_info = experiment.trigger_to_info

    ### Restricting to people or places only
    if cat == 'people':
        restricted_indices = [k for k, v in trigger_to_info.items() if v[1] == 'persona']
        ### Test samples
        eeg = {k : v for k, v in eeg.items() if k in restricted_indices}
        assert len(eeg.keys()) in [16, 20]

    elif cat == 'places':
        restricted_indices = [k for k, v in trigger_to_info.items() if v[1] == 'luogo']
        ### Test samples
        eeg = {k : v for k, v in eeg.items() if k in restricted_indices}
        assert len(eeg.keys()) in [16, 20]

    if args.experiment_id == 'one':
        ### Restricting the data if training/testing on individuals only
        if args.entities == 'individuals_only':
            ### Test samples
            eeg = {k : v for k, v in eeg.items() if k<=100}
            trigger_to_info = {k : v for k, v in trigger_to_info.items() if k <= 100}
        elif args.entities == 'categories_only':
            ### Test samples
            eeg = {k : v for k, v in eeg.items() if k>100}
            trigger_to_info = {k : v for k, v in trigger_to_info.items() if k > 100}
    if args.experiment_id == 'two':
        if args.semantic_category == 'familiar':
            ### Test samples
            eeg = {k : v for k, v in eeg.items() if k<=100}
            trigger_to_info = {k : v for k, v in trigger_to_info.items() if k <= 100}
        elif args.semantic_category == 'famous':
            ### Test samples
            eeg = {k : v for k, v in eeg.items() if k>=100}
            trigger_to_info = {k : v for k, v in trigger_to_info.items() if k >= 100}

    assert len(trigger_to_info.keys()) in [16, 20, 32, 40]

    ### Preparing the input and target samples
    input_samples = list()
    target_samples = list()
    identities = list()

    for trig, erp in eeg.items():

        input_samples.append(erp)

        '''
        if not args.time_resolved:
        else:
            encoding_erp = erp.T
            assert encoding_erp.shape[1] == 128
            input_samples.append(encoding_erp) # ERP in times * electrodes
        '''

        word = trigger_to_info[trig][0] # Converting trigger to word
        if args.word_vectors != 'ceiling':
            word_vector = comp_vectors[word]
        else:
            word_vector = comp_vectors[trig]
        target_samples.append(word_vector)
        word_coarse = trigger_to_info[trig][1]
        if args.experiment_id == 'one':
            word_type = 'entity' if trig <= 100 else 'category'
            word_fine = trigger_to_info[trig][2]
            identities.append((word, word_type, word_coarse, word_fine))
        else:
            word_fine = 'famous' if trig > 100 else 'familiar'
            identities.append((word, '', word_coarse, word_fine))

    ### Crazy PCA
    #pca = sklearn.decomposition.PCA(n_components=0.99)
    #input_samples = pca.fit_transform(input_samples)
    '''
    ### Randomizer
    randomized_indices = random.sample(list(range(len(identities))), k=len(identities))
    #input_samples = [input_samples[i] for i in randomized_indices]
    target_samples = [target_samples[i] for i in randomized_indices]
    #identities = [identities[i] for i in randomized_indices]
    '''

    ### Inverting inputs and targets for encoding
    if args.analysis == 'encoding':
        input_copy = input_samples.copy()
        input_samples = target_samples.copy()
        target_samples = input_copy

    sub_accuracy, word_by_word_evaluation = whole_trial(args, input_samples, \
                                                target_samples, identities, \
                                                #features_per_comb
                                                )
    '''
    if not args.time_resolved:
    else:
        sub_accuracy, word_by_word_evaluation = time_resolved(args, encoding_input, \
                                                    encoding_target, identities)
    '''

    return sub_accuracy, word_by_word_evaluation, times


def whole_trial(args, input_samples, target_samples, \
                        identities, 
                        #features_per_comb
                        ):

    id_wordtype = [t[1] for t in identities]
    id_coarse = [t[2] for t in identities]
    id_fine = [t[3] for t in identities]

    ### All possible combinations
    if '_to_' not in args.entities:
        identities = [k[0] for k in identities]
        ### Pairwise comparisons
        if args.evaluation_method == 'pairwise':
            combs = list(itertools.combinations(list(range(len(target_samples))), r=2))
        elif args.evaluation_method == 'ranking':
            combs = [[i] for i in range(len(target_samples))]
            vectors = list()

    ### Reduced combinations for the transfer task
    else:
        if args.entities == 'all_to_individuals':
            test_indices = [k_i for k_i, k in enumerate(identities) if k[1] == 'entity']
            identities = [k[0] for k in identities]
        elif args.entities == 'individuals_to_categories':
            test_indices = [k_i for k_i, k in enumerate(identities) if k[1] == 'category']
            identities = [k[0] for k in identities]
            
        ### Pairwise comparisons
        if args.evaluation_method == 'pairwise':
            combs = list(itertools.combinations(test_indices, r=2))
        elif args.evaluation_method == 'ranking':
            combs = [[i] for i in test_indices]
            vectors = list()

    identities = [str(ident) for ident in identities]

    accuracies = list()
    word_by_word_evaluation = dict()
    counter = dict()

    for c in tqdm(combs):

        ### Train data
        if args.entities == 'individuals_to_categories':
            train_input = [v for v_i, v in enumerate(input_samples) if \
                                             v_i not in c and v_i not in test_indices]
            train_target = [v for v_i, v in enumerate(target_samples) if \
                                             v_i not in c and v_i not in test_indices]
        else:
            train_input = [v for v_i, v in enumerate(input_samples) if v_i not in c]
            train_target = [v for v_i, v in enumerate(target_samples) if v_i not in c]

        ### Test data
        test_input = [input_samples[v_i] for v_i in c]
        test_target = [target_samples[v_i] for v_i in c]
        test_ids = [identities[v_i] for v_i in c]
        test_coarse = [id_coarse[v_i] for v_i in c]
        test_fine = [id_fine[v_i] for v_i in c]
        test_wordtype = [id_wordtype[v_i] for v_i in c]
        '''
        #if args.feature_reduction not in ['no_reduction']:
        if isinstance(features_per_comb, dict):
            ### Feature selection if required

            test_ids = tuple(test_ids)
            if test_ids in list(features_per_comb.keys()):
                selected_indices = features_per_comb[test_ids]
            else:
                inverse_ids = (test_ids[1], test_ids[0])
                assert inverse_ids in list(features_per_comb.keys())
                selected_indices = features_per_comb[inverse_ids]

            if args.analysis == 'decoding':

                train_input = [vec[selected_indices] for vec in train_input] 
                test_input = [vec[selected_indices] for vec in test_input] 

            elif args.analysis == 'encoding':
                train_target = [[vec[i] for i in selected_indices] for vec in train_target] 
                test_target = [[vec[i] for i in selected_indices] for vec in test_target] 
        '''


        #selected_indices = [int(v) for v in random.choices(population=range(5000), k=100)]
        #train_input = [vec[:1000] for vec in train_input] 
        #test_input = [vec[:1000] for vec in test_input] 

        ### Loading the classifier
        #classifier = LinearRegression()
        classifier = Ridge()
        classifier.fit(numpy.array(train_input), numpy.array(train_target))
        
        ### Predicting
        predictions = classifier.predict(test_input)

        if args.evaluation_method == 'pairwise':
            accuracies, word_by_word_evaluation, counter = pairwise_accuracy(\
                                      args, predictions, word_by_word_evaluation, \
                                      test_target, test_ids, test_coarse, \
                                      test_fine, test_wordtype, accuracies, \
                                      counter)

        elif args.evaluation_method == 'ranking':
            assert predictions.shape[0] == 1
            vectors.append(predictions[0, :])

    if args.evaluation_method == 'ranking':
        #if 'transfer' in args.analysis:
        if '_to_' in args.entities:
            target_samples = [s for s_i, s in enumerate(target_samples) if s_i in test_indices]
            reduced_identities = [k for k_i, k in enumerate(identities) if k_i in test_indices]
            accuracies, word_by_word_evaluation = ranking_accuracy(vectors, target_samples, \
                                                   reduced_identities, word_by_word_evaluation, args)
        else:
            accuracies, word_by_word_evaluation, counter = ranking_accuracy(vectors, \
                    target_samples, identities, word_by_word_evaluation, args)

    #if args.entities == 'individuals_to_categories':
        #max_evaluations = len(test_indices)-1
    #else:
        #max_evaluations = len(identities)-1
    word_by_word_evaluation = {k : v/counter[k] for k, v in word_by_word_evaluation.items()}

    '''
    if args.entities == 'individuals_to_categories':
        assert len(word_by_word_evaluation.keys()) == len(test_indices)
    else:
        assert len(word_by_word_evaluation.keys()) == len(input_samples)
    '''

    #if args.evaluation_method == 'pairwise':
    #elif args.evaluation_method == 'ranking':
    #    sub_accuracy = numpy.median(accuracies)
    sub_accuracy = numpy.average(accuracies)
    #print('median: {}'.format(numpy.median(accuracies)))
    print('average: {}'.format(numpy.average(accuracies)))
    #print('mode: {}'.format(stats.mode(accuracies)[0][0]))
    #print('skewness: {}'.format(stats.skew(accuracies)))
    #print('test of normality: {}'.format(stats.normaltest(accuracies)))

    return sub_accuracy, word_by_word_evaluation

def ranking_accuracy(vectors, target_samples, identities, word_by_word_evaluation, args):

    accuracies = list()
    counter = dict()
    
    for vec_i, vec in enumerate(vectors):

        results = list()

        ### Computing the accuracy
        for targ_id, targ_vec in enumerate(target_samples):
            sim = stats.spearmanr(vec, targ_vec)[0]
            #sim = 1 - scipy.spatial.distance.cosine(vec, targ_vec)
            #sim = stats.pearsonr(vec, targ_vec)[0]
            results.append((targ_id, sim))
        
        sorted_results = sorted(results, key=lambda item : item[1], \
                                reverse=True)
        rank = [i+1 for i, res in enumerate(sorted_results) if res[0]==vec_i]
        assert len(rank) == 1
        acc = 1 - ((rank[0] - 1) / (len(vectors) -1 ))
        accuracies.append(acc)
        
        test_id = identities[vec_i]

        word_by_word_evaluation[test_id] = acc
        counter[test_id] = 1
        '''
        if not args.time_resolved:
            word_by_word_evaluation[test_id] = acc
        else:
            if test_id not in word_by_word_evaluation.keys():
                word_by_word_evaluation[test_id] = [acc]
            else:
                word_by_word_evaluation[test_id].append(acc)
        '''

    return accuracies, word_by_word_evaluation, counter

def pairwise_accuracy(args, predictions, word_by_word_evaluation, \
                      test_target, test_ids, test_coarse, \
                      test_fine, test_wordtype, accuracies, counter):
        
    ### Computing the accuracy
    #wrong = stats.spearmanr(predictions[0], test_target[1])[0] * 2
    wrong_one = stats.spearmanr(predictions[0], test_target[1])[0]
    wrong_two = stats.spearmanr(predictions[1], test_target[0])[0]
    wrong = wrong_one + wrong_two

    correct = 0.

    for i in range(2):

        correct += stats.spearmanr(predictions[i], test_target[i])[0]

    ### Counting occurrences

    ### Experiment one
    if args.experiment_id == 'one':
        if 'category' not in test_wordtype:

            ### coarse
            test_cat = '_'.join(['entity'] + sorted(test_coarse))
            if test_cat not in counter.keys():
                counter[test_cat] = 1
            else:
                counter[test_cat] += 1
            ### fine
            test_cat = '_'.join(['entity'] + sorted(test_fine))
            if test_cat not in counter.keys():
                counter[test_cat] = 1
            else:
                counter[test_cat] += 1
        elif 'entity' not in test_wordtype:
            ### coarse
            test_cat = '_'.join(['category'] + sorted(test_coarse))
            if test_cat not in counter.keys():
                counter[test_cat] = 1
            else:
                counter[test_cat] += 1
            ### fine
            test_cat = '_'.join(['category'] + sorted(test_fine))
            if test_cat not in counter.keys():
                counter[test_cat] = 1
            else:
                counter[test_cat] += 1
        else:
            ### coarse
            test_cat = '_'.join(['mixed'] + sorted(test_coarse))
            if test_cat not in counter.keys():
                counter[test_cat] = 1
            else:
                counter[test_cat] += 1
            ### fine
            test_cat = '_'.join(['mixed'] + sorted(test_fine))
            if test_cat not in counter.keys():
                counter[test_cat] = 1
            else:
                counter[test_cat] += 1
    else:
        ### coarse
        test_cat = '_'.join(sorted(test_coarse))
        if test_cat not in counter.keys():
            counter[test_cat] = 1
        else:
            counter[test_cat] += 1
        ### fine
        test_cat = '_'.join(sorted(test_fine))
        if test_cat not in counter.keys():
            counter[test_cat] = 1
        else:
            counter[test_cat] += 1
        ### both
        test_cat = '_'.join(sorted(test_coarse) + sorted(test_fine))
        if test_cat not in counter.keys():
            counter[test_cat] = 1
        else:
            counter[test_cat] += 1

    for i in range(2):

        test_id = test_ids[i]
        if test_id not in counter.keys():
            counter[test_id] = 1
        else:
            counter[test_id] += 1

    if correct > wrong:
        accuracies.append(1)

        ### Recording the correctly classified performance for the word
        for i in range(2):

            test_id = test_ids[i]
            if test_id in word_by_word_evaluation.keys():
                word_by_word_evaluation[test_id] += 1
            else:
                word_by_word_evaluation[test_id] = 1

        ### Recording the correctly classified performance 
        ### for the coarse category

        ### Experiment one

        if args.experiment_id == 'one':
            ### Entity-only
            if 'category' not in test_wordtype:

                ### coarse
                test_cat = '_'.join(['entity'] + sorted(test_coarse))
                if test_cat in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] += 1
                else:
                    word_by_word_evaluation[test_cat] = 1
                ### fine
                test_cat = '_'.join(['entity'] + sorted(test_fine))
                if test_cat in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] += 1
                else:
                    word_by_word_evaluation[test_cat] = 1

            ### Category-only
            elif 'entity' not in test_wordtype:

                ### coarse
                test_cat = '_'.join(['category'] + sorted(test_coarse))
                if test_cat in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] += 1
                else:
                    word_by_word_evaluation[test_cat] = 1
                ### fine
                test_cat = '_'.join(['category'] + sorted(test_fine))
                if test_cat in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] += 1
                else:
                    word_by_word_evaluation[test_cat] = 1

            ### Entity vs category
            else:

                ### coarse
                test_cat = '_'.join(['mixed'] + sorted(test_coarse))
                if test_cat in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] += 1
                else:
                    word_by_word_evaluation[test_cat] = 1
                ### fine
                test_cat = '_'.join(['mixed'] + sorted(test_fine))
                if test_cat in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] += 1
                else:
                    word_by_word_evaluation[test_cat] = 1
        else:
            ### coarse
            test_cat = '_'.join(sorted(test_coarse))
            if test_cat in word_by_word_evaluation.keys():
                word_by_word_evaluation[test_cat] += 1
            else:
                word_by_word_evaluation[test_cat] = 1
            ### fine
            test_cat = '_'.join(sorted(test_fine))
            if test_cat in word_by_word_evaluation.keys():
                word_by_word_evaluation[test_cat] += 1
            else:
                word_by_word_evaluation[test_cat] = 1
            ### both
            test_cat = '_'.join(sorted(test_coarse) + sorted(test_fine))
            if test_cat in word_by_word_evaluation.keys():
                word_by_word_evaluation[test_cat] += 1
            else:
                word_by_word_evaluation[test_cat] = 1

    else:
        accuracies.append(0)
        ### Adding the word to the keys of the word-by-word dictionary
        for i in range(2):

            test_id = test_ids[i]
            if test_id not in word_by_word_evaluation.keys():
                word_by_word_evaluation[test_id] = 0
        if args.experiment_id == 'one':
            ### Entity-only
            if 'category' not in test_wordtype:

                ### coarse
                test_cat = '_'.join(['entity'] + sorted(test_coarse))
                if test_cat not in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] = 0
                ### fine
                test_cat = '_'.join(['entity'] + sorted(test_fine))
                if test_cat not in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] = 0

            ### Category-only
            elif 'entity' not in test_wordtype:

                ### coarse
                test_cat = '_'.join(['category'] + sorted(test_coarse))
                if test_cat not in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] = 0
                ### fine
                test_cat = '_'.join(['category'] + sorted(test_fine))
                if test_cat not in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] = 0

            ### Entity vs category
            else:

                ### coarse
                test_cat = '_'.join(['mixed'] + sorted(test_coarse))
                if test_cat not in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] = 0
                ### fine
                test_cat = '_'.join(['mixed'] + sorted(test_fine))
                if test_cat not in word_by_word_evaluation.keys():
                    word_by_word_evaluation[test_cat] = 0
        else:
            ### coarse
            test_cat = '_'.join(sorted(test_coarse))
            if test_cat not in word_by_word_evaluation.keys():
                word_by_word_evaluation[test_cat] = 0
            ### fine
            test_cat = '_'.join(sorted(test_fine))
            if test_cat not in word_by_word_evaluation.keys():
                word_by_word_evaluation[test_cat] = 0
            ### both
            test_cat = '_'.join(sorted(test_coarse) + sorted(test_fine))
            if test_cat not in word_by_word_evaluation.keys():
                word_by_word_evaluation[test_cat] = 0

    return accuracies, word_by_word_evaluation, counter

'''
def time_resolved(args, input_samples, target_samples, identities):

    if 'transfer' not in args.analysis:
        identities = [k[0] for k in identities]
        ### Pairwise comparisons
        if args.evaluation_method == 'pairwise':
            combs = list(itertools.combinations(list(range(len(target_samples))), r=2))
        elif args.evaluation_method == 'ranking':
            combs = [[i] for i in range(len(target_samples))]
    else:
        test_indices = [k_i for k_i, k in enumerate(identities) if k[1] == 'category']
        identities = [k[0] for k in identities]
        ### Pairwise comparisons
        if args.evaluation_method == 'pairwise':
            combs = list(itertools.combinations(test_indices, r=2))
        elif args.evaluation_method == 'ranking':
            combs = [[i] for i in test_indices]

    subject_accuracies = list()
    word_by_word_evaluation = dict()
    n_times = target_samples[0].shape[0]

    for t in tqdm(range(n_times)):
        vectors = list()
        
        for c in combs:

            ### Train data
            if 'transfer' not in args.analysis:
                train_input = [v for v_i, v in enumerate(input_samples) if v_i not in c]
                train_target = [v[t, :] for v_i, v in enumerate(target_samples) if v_i not in c]
            else:
                train_input = [v for v_i, v in enumerate(input_samples) if \
                                                 v_i not in c and v_i not in test_indices]
                train_target = [v[t, :] for v_i, v in enumerate(target_samples) if \
                                                 v_i not in c and v_i not in test_indices]
            
            ### Test data
            test_input = [input_samples[v_i] for v_i in c]
            test_target = [target_samples[v_i][t, :] for v_i in c]
            test_ids = [identities[v_i] for v_i in c]
            
            ### Loading the classifier
            classifier = Pipeline([('pca', PCA(n_components=0.9)), (' ridge', Ridge())])
            classifier.fit(train_input, train_target)
            
            ### Predicting
            predictions = classifier.predict(test_input)
            
            if args.evaluation_method == 'pairwise':
                accuracies, word_by_word_evaluation = pairwise_accuracy(\
                                                      predictions, \
                                                      word_by_word_evaluation, \
                                                      test_target, test_ids, \
                                                      accuracies, counter)
            elif args.evaluation_method == 'ranking':
                assert predictions.shape[0] == 1
                vectors.append(predictions[0, :])

        if args.evaluation_method == 'ranking':
            if 'transfer' in args.analysis:
                current_target = [v[t, :] for v_i, v in enumerate(target_samples) if v_i in test_indices]
                reduced_identities = [k for k_i, k in enumerate(identities) if k_i in test_indices]
                accuracies, word_by_word_evaluation = ranking_accuracy(vectors, current_target, \
                                                       reduced_identities, word_by_word_evaluation, args)
            else:
                accuracies, word_by_word_evaluation = ranking_accuracy(vectors, current_target, \
                                                       identities, word_by_word_evaluation, args)

        t_accuracy = numpy.average(accuracies)
        subject_accuracies.append(t_accuracy)

    if 'transfer' not in args.analysis:
        max_evaluations = len(identities)-1
    else:
        max_evaluations = len(test_indices)-1
    max_evaluations = sum([1 for c in combs if 0 in c])*n_times
    word_by_word_evaluation = {k : [value/max_evaluations for value in v] for k, v in word_by_word_evaluation.items()}

    if 'transfer' not in args.analysis:
        assert len(word_by_word_evaluation.keys()) == len(input_samples)
    else:
        assert len(word_by_word_evaluation.keys()) == len(test_indices)

    return subject_accuracies, word_by_word_evaluation
'''
