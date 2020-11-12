import sklearn
import scipy
import numpy
import pickle
import os
import collections
import random
import torch
import logging
from scipy.stats.morestats import wilcoxon

from utils.neural_networks import train_torch_net

def rsa(args, list_of_vectors):
    output_set = []
    for vector_one in list_of_vectors:
        rsa_vector = numpy.zeros(len(list_of_vectors))
        for sim_index, vector_two in enumerate(list_of_vectors): 
            if args.distance_measure == 'euclidean_distance':
                rsa_vector[sim_index] = cos(vec_one, vec_two)
            elif args.distance_measure == 'spearman_correlation':
                rsa_vector[sim_index] = float(scipy.stats.spearmanr(vector_one, vector_two)[0])
        output_set.append(rsa_vector)
    return(output_set)

def get_details(folder):
    with open(os.path.join(folder, 'README.txt'), 'r') as f:
        l = f.readlines()
        subjects = int(l[0].split('\t')[1])
        runs = int(l[1].split('\t')[1])
    return subjects, runs

def print_wilcoxon(args, s, model_path, good_one, bad_two):
    logging.info('Subject {}'.format(s))
    for test_key, results_one in good_one.items():
        logging.info('Test: {}'.format(test_key))
        results_two = bad_two[test_key]
        average_one = numpy.average(results_one)
        median_one = numpy.median(results_one)
        average_two = numpy.average(results_two)
        median_two = numpy.median(results_two)
        z_value, p_value = wilcoxon(results_one, results_two)
        effect_size = abs(z_value / math.sqrt(len(results_one)))
        if args.write_to_file:
            with open(os.path.join(model_path, 'classification_results.txt'), 'a') as o:

                o.write('Subject {} -Test: {}\n\nGood vs bad pairings\nAverage: {} vs {}\nMedian: {} vs {}\nSignificance testing p-value and effect size: {}, {}\n\n\n'.format(s, test_key.capitalize(), average_one, average_two, median_one, median_two, p_value, effect_size))
        else:
            logging.info('Average: good {} vs {}\nMedian: good {} vs {}\nSignificance testing p-value and effect size: {}, {}\n'.format(average_one, average_two, median_one, median_two, p_value, effect_size))

def print_results(args, s, model_path, results, final_results):
    for test_key, test_results in results.items():
        average_result = numpy.average(test_results)
        median_result = numpy.median(test_results)
        final_results[test_key].append(average_result)
        print('\t{} vectors - Average and median accuracy results across {} evaluations: {}, {}'.format(test_key.capitalize(), len(test_results), average_result, median_result))
        if args.write_to_file:
            with open(os.path.join(model_path, 'classification_results.txt'), 'a') as o:

                o.write('{} vectors\n\nAverage and median accuracy results across {} evaluations: {}, {}\n\n\n'.format(test_key.capitalize(), len(test_results), average_result, median_result))

#def load_word_vectors(vector_model, data_type, data_category, men=False, categories=False):
def load_word_vectors(vector_model, men=False, categories=False):
    if men:
        if vector_model == 'transe':
            vector_model = 'wiki2vec'
        if vector_model == 'ernie':
            vector_model = 'bert_two'
        with open('word_vectors/{}_men.vec'.format(vector_model)) as input_file:
            lines = [l.strip().split('\t') for l in input_file.readlines()]
        word_vectors = {line[0] : numpy.asarray(line[1:], dtype=numpy.single) for line in lines}
        output_dimensions = len([v for k, v in word_vectors.items()][0])
    else:
        #data_type = 'uk' if categories else 'individuals'
        #with open('word_vectors/{}_{}_{}.vec'.format(vector_model, data_type, data_category)) as input_file:
        with open('word_vectors/{}_individuals.vec'.format(vector_model)) as input_file:
            lines = [l.strip().split('\t') for l in input_file.readlines()]
        word_vectors = {line[0] : numpy.asarray(line[1:], dtype=numpy.single) for line in lines}
        output_dimensions = len([v for k, v in word_vectors.items()][0])

    return word_vectors, output_dimensions


def train_test_split(args, first_pass, second_pass, first_shuffled_pass, second_shuffled_pass, first_randomized_pass, second_randomized_pass, average_pass, first_men_pass, second_men_pass, test_items, test_instance):

    train_dict = collections.defaultdict(list)
    test_dict = collections.defaultdict(list)

    # Preparing the train data, so as to have two experimental conditions (a true condition and a control, shuffled condition)

    # True condition
    train_one = first_pass.copy()
    if args.blind_test: 
        train_one = [l for l in first_pass if l[2] != test_instance[2]]
        test_one = [l for l in first_pass if l[2] == test_instance[2]]
    train_two = [l for instance_index, l in enumerate(second_pass) if instance_index not in test_items]
    train_dict['original'] = train_one + train_two

    # Shuffled dimensions condition

    train_shuffled_one = first_shuffled_pass.copy()
    train_shuffled_two = [l for instance_index, l in enumerate(second_shuffled_pass) if instance_index not in test_items]
    train_dict['shuffled'] = train_shuffled_one + train_shuffled_two

    # Averaged condition

    train_dict['averaged'] = [l for instance_index, l in enumerate(average_pass) if instance_index not in test_items]

    # Randomized assignment control condition

    train_random_one = first_randomized_pass.copy()
    train_random_two = [l for instance_index, l in enumerate(second_randomized_pass) if instance_index not in test_items]

    train_dict['randomized'] = train_random_one + train_random_two

    # MEN condition
    train_men_one = first_men_pass.copy()
    train_men_two = [l for instance_index, l in enumerate(second_men_pass) if instance_index not in test_items]
    train_dict['men'] = train_men_one + train_men_two

    # Ranking test item
    if len(test_items) == 1:

        if not args.blind_test:
            test_dict['original'] = (second_pass[test_items[0]][0], second_pass[test_items[0]][2])
        else:
            test_dict['original'] = [[second_pass[test_items[0]][0], second_pass[test_items[0]][2]]] + [[k[0], k[2]] for k in test_one]

        test_dict['men'] = (second_men_pass[test_items[0]][0], second_men_pass[test_items[0]][2])

        test_dict['shuffled'] = (second_shuffled_pass[test_items[0]][0], second_shuffled_pass[test_items[0]][2])

        test_dict['averaged'] = (average_pass[test_items[0]][0], average_pass[test_items[0]][2])

        test_dict['randomized'] = (second_randomized_pass[test_items[0]][0], second_randomized_pass[test_items[0]][2])
    # Pairwise test items
    if len(test_items) == 2:

        test_dict['original'] = [[second_pass[test_items[0]][0], second_pass[test_items[0]][1], second_pass[test_items[0]][2]], [second_pass[test_items[1]][0], second_pass[test_items[1]][1], second_pass[test_items[1]][2]]]

        test_dict['men'] = [[second_men_pass[test_items[0]][0], second_men_pass[test_items[0]][1], second_men_pass[test_items[0]][2]], [second_men_pass[test_items[1]][0], second_men_pass[test_items[1]][1], second_men_pass[test_items[1]][2]]]

        test_dict['shuffled'] = [[second_shuffled_pass[test_items[0]][0], second_shuffled_pass[test_items[0]][1], second_shuffled_pass[test_items[0]][2]], [second_shuffled_pass[test_items[1]][0], second_shuffled_pass[test_items[1]][1], second_shuffled_pass[test_items[1]][2]]]

        test_dict['averaged'] = [[average_pass[test_items[0]][0], average_pass[test_items[0]][1], average_pass[test_items[0]][2]], [average_pass[test_items[1]][0], average_pass[test_items[1]][1], average_pass[test_items[1]][2]]]

        test_dict['randomized'] = [[second_randomized_pass[test_items[0]][0], second_randomized_pass[test_items[0]][1], second_randomized_pass[test_items[0]][2]], [second_randomized_pass[test_items[1]][0], second_randomized_pass[test_items[1]][1], second_randomized_pass[test_items[1]][2]]]

        #test_dict['shuffled'] = [[second_pass[test_items[0]][0], train_random_one[test_items[0]][1], train_random_one[test_items[0]][2]], [second_pass[test_items[1]][0], train_random_one[test_items[1]][1], train_random_one[test_items[1]][2]]]

    return train_dict, test_dict

def get_predictions(args, model, train_data, test_data):

    if args.model == 'ridge':

        # Train data
        model.fit([item[0] for item in train_data], [item[1] for item in train_data])

        #Test data
        if len(test_data) != 2:
            predictions = model.predict(test_data.reshape(1, -1))[0]
        if len(test_data) == 2:
            predictions = []
            for test_item in test_data:
                predictions.append(model.predict(numpy.asarray(test_item, dtype=numpy.single).reshape(1, -1)))

    if args.model == 'perceptron':

        # Train data
        train_torch_net(model, [item[0] for item in train_data], [item[1] for item in train_data])

        # Test data
        #test_data = [k for k in test_data if k != 0.0]
        if type(test_data) != 'list':
            test_tensor = torch.tensor(test_data, device=model[2]).float().view(1, model[0].input_dimensions)
            predictions = model[0](test_tensor).cpu().detach().numpy()[0]
        if len(test_data) == 2:
            test_tensor = torch.tensor(test_data, device=model[2]).float().view(2, model[0].input_dimensions)
            predictions = model[0](test_tensor).cpu().detach().numpy()

    if args.model == 'cnn':

        # Train data
        cnn_train_brain = [torch.tensor(item[0].get_fdata().tolist(), device=model[2]).view(model[3]) for item in train_data]
        cnn_train_embeddings = [torch.tensor(item[1], device=model[2]).view(1, model[4]) for item in train_data]
        train_torch_net(model, cnn_train_brain, cnn_train_embeddings)

        # Test data
        if type(test_data) != 'list':
            cnn_test_brain = torch.tensor(test_data.get_fdata().tolist(), device=model[2]).view(model[3])
            predictions = model[0](cnn_test_brain).cpu().detach().numpy()[0]
        else:
            predictions = []
            for test_item in test_data:
                cnn_test_brain = torch.tensor(test_item.get_fdata().tolist(), device=model[2]).view(model[3])
                predictions.append(model[0](cnn_test_brain).cpu().detach().numpy())

    return predictions
