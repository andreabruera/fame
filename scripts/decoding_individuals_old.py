import nilearn
import re
import argparse
import logging
import os
import collections
import numpy
import math
import sklearn
import scipy
import itertools
import wikipedia2vec
import random
import matplotlib
import pickle
import torch

from tqdm import tqdm
from nilearn import plotting, image
from nilearn.plotting import plot_stat_map, show
from nilearn.input_data import NiftiMasker
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from scipy.spatial import distance
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt

from utils.neural_networks import torch_linear, torch_CNN, train_torch_net, torch_weights_init 
from utils.general_utilities import cos, rsa, get_details, print_wilcoxon, print_results, load_word_vectors, train_test_split, get_predictions
from utils.brain_images import get_brain_images, find_stable_voxels, get_brain_areas
from utils.load_eeg import load_eeg_vectors, get_eeg_words 
#from utils.voxel_analysis import voxel_analysis
#from utils.category_decoding import category_decoding

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, required = True, choices = ['mmms_fame', 'ltm_fame', 'eeg_stanford'], help = 'Name of the dataset, which will also be the name of the original folder and of the plots folder')
parser.add_argument('--vectors', type = str, choices = ['wiki2vec', 'transe', 'ernie', 'bert', 'bert_two', 'word2vec_two', 'word2vec'], default = 'wiki2vec', help = 'Computational vectors to be used for the regression')
parser.add_argument('--brain_modality', type = str, choices = ['fmri', 'eeg'], default = 'fmri', help = 'Brain data modality to use for testing')
parser.add_argument('--tr', type = int, default = 2, help = 'TRs for the brain images')
parser.add_argument('--amount_stable_voxels', type = int, default = 5000, help = 'The amount of stable voxels to be retained for regression')
parser.add_argument('--unsmoothed', action = 'store_true', required = False, help = 'Indicate whether to use a smoothed version of the brain data or not')
parser.add_argument('--subtraction', action = 'store_true', required = False, help = 'Indicates whether to subtract the fixation-cross average image or not')
parser.add_argument('--significance_test', action = 'store_true', required = False, help = 'Indicates whether to carry out the significance testing or not')
parser.add_argument('--find_stable_voxels', action = 'store_true', required = False, help = 'Indicates whether to select the most stable voxels or not')
parser.add_argument('--write_to_file', action = 'store_true', required = False, help = 'Indicates whether to write the results to file or not')
parser.add_argument('--voxel_analysis', action = 'store_true', required = False, help = 'Indicates whether to conduct voxel analysis to find the most activated areas or not')
parser.add_argument('--blind_test', action = 'store_true', default = False, help = 'Indicates whether to use one of the two brain images for a certain individual to the learning algorithm before testing or not')
parser.add_argument('--distance_measure', type = str, required = False, default = 'spearman_correlation', choices = ['euclidean_distance', 'spearman_correlation'], help = 'Name of the distance measure to be used throughout the experiment')
parser.add_argument('--evaluation_method', type = str, required = False, default = 'ranking', choices = ['pairwise', 'ranking'], help = 'Indicates whether to use the pairwise or the ranking evaluation metric')
parser.add_argument('--model', type = str, required = False, default = 'ridge', choices = ['cnn', 'ridge', 'rsa', 'perceptron'], help = 'Indicates which model to use for regression')
parser.add_argument('--cuda_device', type = int, required = False, default = '0', choices = [0, 1, 2], help = 'Indicates which GPU to use for the neural network')
parser.add_argument('--dump_pickled_subjects', action='store_true', default=False, help='Indicates whether to dump pickled subjects or not')
parser.add_argument('--load_pickled_subjects', action='store_true', default=True, help='Indicates whether to load pickled subjects or not')
parser.add_argument('--direction', choices=['decoding', 'encoding'], default='decoding', help='Indicates which analysis to carry out')
args = parser.parse_args()

folder = os.path.join('neuro_data', args.dataset)
if args.dataset != 'eeg_stanford':
    subjects, runs = get_details(folder)
else:
    subjects = 10
    runs = 1

masker = NiftiMasker(mask_strategy='template', detrend = True, high_pass = 0.005, t_r = 2)
sample_img = image.load_img('/import/cogsci/andrea/github/fame/neuro_data/mmms_fame/sub_01/func/run_02/4D_smooth_run_02.nii')
sample_slice = image.index_img(sample_img, 100)
input_shape = [1, 1] + [k for k in sample_slice.get_fdata().shape]
masked_img = masker.fit_transform(sample_img)



if args.find_stable_voxels:
    input_dimensions = args.amount_stable_voxels
else:
    input_dimensions = 68049

# Acquiring the word vectors

if args.dataset == 'mmms_fame':
    word_vectors, output_dimensions = load_word_vectors(args.vectors)
    men_vectors_and_names, _ = load_word_vectors(args.vectors, men=True)
    men_vectors = [v for k, v in men_vectors_and_names.items()]
elif args.dataset == 'eeg_stanford':
    word_vectors, output_dimensions = load_word_vectors('bert_eeg_stanford')
else:
    output_dimensions = 2
    word_vectors = defaultdict(list)

# Preparing the neural networks
        
cuda = torch.device('cuda:{}'.format(args.cuda_device))
torch.manual_seed(0)
torch.cuda.manual_seed(0)
loss_function = torch.nn.CosineEmbeddingLoss()

if args.model == 'perceptron':
    torch_perceptron = torch_linear(output_dimensions, input_dimensions).float()
    torch_perceptron = torch_perceptron.to(device=cuda)
    torch_perceptron.zero_grad()
    model = [torch_perceptron, loss_function, cuda]
    models = [model]
if args.model == 'cnn':
    torch_cnn = torch_CNN(input_shape, output_dimensions).float()
    torch_cnn = torch_cnn.to(device=cuda)
    torch_cnn.zero_grad()
    model = [torch_cnn, loss_function, cuda, input_shape, output_dimensions]
    models = [model]
    args.find_stable_voxels = False
    args.amount_stable_voxels = 68049
if args.model == 'rsa':
    models = ['rsa']
if args.model == 'ridge':
    model = Ridge(alpha=1.0)
    models = [model]

if args.evaluation_method == 'pairwise':
    logging.info('Now moving on to the pairwise evaluation...')
if args.evaluation_method == 'ranking':
    logging.info('Now starting the ranking accuracy evaluation...')
logging.info('Now performing classification evaluation for model: {}'.format(args.model))


final_results = defaultdict(list)
histogram_results = defaultdict(list)
people_results = defaultdict(list)
     
for s in range(1, subjects+1):

    # Loading the pickle with the brain images, or collecting the brain images across all the runs    
    if args.load_pickled_subjects:
        if args.dataset == 'eeg_stanford':
            sub_images_individuals = load_eeg_vectors(s)
        elif args.brain_modality == 'fmri':
            sub_images_individuals = pickle.load(open('pickles/fmri_sub_{:02}.pkl'.format(s), 'rb'))
        elif args.brain_modality == 'eeg':
            #sub_images_individuals = pickle.load(open('pickles/eeg_images/sub-{:02}_eeg_vecs.pkl'.format(s), 'rb'))
            sub_images_individuals = pickle.load(open('/import/cogsci/andrea/dataset/eeg_images_new/sub-{:02}_eeg_vecs.pkl'.format(s), 'rb'))

    else:
        sub_images_individuals = get_brain_images(args)

    if args.find_stable_voxels:
        sub_images_individuals, masked_indices = find_stable_voxels(args.amount_stable_voxels, sub_images_individuals)
    elif args.dataset != 'eeg_stanford':
        sub_images_individuals = {k : v for k, v in sub_images_individuals.items() if len(v) == 2}

    if args.brain_modality == 'fmri' and args.dataset != 'eeg_stanford':
        get_brain_areas(s, masker, masked_indices)

    if args.model == 'cnn':
        individuals_copy_for_cnn = sub_images_individuals.copy()
        sub_images_individuals = defaultdict(list)
        for k, v in individuals_copy_for_cnn.items():
            for vec in v:
                cnn_image = masker.inverse_transform(vec)
                sub_images_individuals[k].append(cnn_image)

    for m in models:
        model_name = re.sub('\(.+', '', str(m), flags=re.DOTALL)
        model_path = os.path.join('plots', args.dataset, 'sub_{:02}'.format(s), model_name)
        os.makedirs(model_path, exist_ok = True)
        collection = []
        results = []

        # RSA-based decoding

        if model_name == 'rsa':

            results = defaultdict(list)
            good_results = defaultdict(list)
            wrong_results = defaultdict(list)
            test_set = defaultdict(list)
            fmri_images = []
            rsa_identities = []
            for k, v in sub_images_individuals.items():
                try:
                    fmri_images.append(v[1])
                except IndexError:
                    fmri_images.append(v[0]) 
                test_set['rsa_individuals'].append(word_vectors[k])
                rsa_identities.append(k) 
            #test_set['rsa_shuffled'] = [numpy.asarray(random.sample(instance.tolist(), k = len(instance.tolist()))) for instance in test_set['rsa_individuals']]
            test_set['rsa_shuffled'] = random.sample(test_set['rsa_individuals'], k=len(test_set['rsa_individuals']))
            fmri_rsa_full = rsa(args, fmri_images)
            #embeddings_rsa_full = rsa(args, test_data)
            embeddings_rsa_full = rsa(args, test_set['rsa_individuals'])


            '''

            pairing_combinations= [k for k in itertools.combinations(range(len(fmri_rsa_full)), 2)]
            for test_key, test_data in tqdm(test_set.items()):
                embeddings_rsa_full = rsa(args, test_data)

                if args.evaluation_method == 'pairwise':
                    for combination in pairing_combinations:
                        fmri_rsa = numpy.delete(fmri_rsa_full, combination, axis=1)
                        embeddings_rsa = numpy.delete(embeddings_rsa_full, combination, axis = 1)
                        assert len(fmri_rsa[0]) == len(fmri_rsa_full[0]) - 2
                        if args.distance_measure == 'euclidean_distance':
                            good_sim_one = cosine_similarity(embeddings_rsa[combination[0]].reshape(1, -1), fmri_rsa[combination[0]].reshape(1, -1))
                            good_sim_two = cosine_similarity(embeddings_rsa[combination[1]].reshape(1, -1), fmri_rsa[combination[1]].reshape(1, -1))
                            wrong_sim_one = cosine_similarity(embeddings_rsa[combination[0]].reshape(1, -1), fmri_rsa[combination[1]].reshape(1, -1))
                            wrong_sim_two = cosine_similarity(embeddings_rsa[combination[1]].reshape(1, -1), fmri_rsa[combination[0]].reshape(1, -1))
                        elif args.distance_measure == 'spearman_correlation':
                            good_sim_one = spearmanr(embeddings_rsa[combination[0]], fmri_rsa[combination[0]])
                            good_sim_two = spearmanr(embeddings_rsa[combination[1]], fmri_rsa[combination[1]])
                            wrong_sim_one = spearmanr(embeddings_rsa[combination[0]], fmri_rsa[combination[1]])
                            wrong_sim_two = spearmanr(embeddings_rsa[combination[1]], fmri_rsa[combination[0]])
                        good_pairing = good_sim_one + good_sim_two
                        wrong_pairing = wrong_sim_one + wrong_sim_two
                        if args.significance_test:
                            good_results[test_key].append(float(good_pairing[0]))
                            wrong_results[test_key].append(float(wrong_pairing[0]))
                        if good_pairing > wrong_pairing:
                            results[test_key].append(float(1))
                            histogram_results[test_key].append(float(1))
                        else:
                            results[test_key].append(float(0))
                            histogram_results[test_key].append(float(0))

                if args.evaluation_method == 'ranking':
                    for embeddings_index, person_full in enumerate(embeddings_rsa_full):
                        fmri_rsa = numpy.delete(fmri_rsa_full, embeddings_index, axis=1)
                        embeddings_rsa = numpy.delete(embeddings_rsa_full, embeddings_index, axis = 1)
                        embeddings_person = embeddings_rsa[embeddings_index]
                        assert len(fmri_rsa[0]) == len(fmri_rsa_full[0]) - 1
                        assert len(embeddings_rsa[0]) == len(embeddings_rsa_full[0]) - 1
                        if args.distance_measure == 'euclidean_distance':
                            sims = {fmri_index : cosine_similarity(embeddings_person.reshape(1, -1), fmri_person.reshape(1, -1)) for fmri_index, fmri_person in enumerate(fmri_rsa)}  
                        if args.distance_measure == 'spearman_correlation':
                            sims = {fmri_index : spearmanr(embeddings_person, fmri_person) for fmri_index, fmri_person in enumerate(fmri_rsa)}  
                        rankings = {fmri_index : sims[fmri_index] for fmri_index in sorted(sims, key = sims.get, reverse = True)}
                        rank = 0
                        for fmri_index, sim in rankings.items():
                            rank += 1
                            if fmri_index == embeddings_index:
                                break
                        accuracy = 1 - ((rank - 1) / (len(embeddings_rsa) - 1))
                        results[test_key].append(accuracy)
                        histogram_results[test_key].append(accuracy)

            print('\tSubject {}'.format(s))
            print_results(args, s, model_path, results, final_results)
            if args.significance_test:
                print_wilcoxon(args, s, model_path, good_results, wrong_results)
            '''

        # Decoding of individual people representations to vectors

        if args.model == 'ridge' or args.model == 'perceptron' or args.model == 'cnn':

            results = defaultdict(list)

            # Images for the 'original' condition

            if args.dataset != 'eeg_stanford':
                first_pass = [[v[0], word_vectors[k], k] for k, v in sub_images_individuals.items()]
                second_pass = [[v[1], word_vectors[k], k] for k, v in sub_images_individuals.items()]
            else:
                first_pass = [[v[i], word_vectors[k], k] for k, v in sub_images_individuals.items() for i in range(7)]
                second_pass = [[v[7], word_vectors[k], k] for k, v in sub_images_individuals.items()]

            # Images for the 'shuffled' condition

            if args.dataset != 'eeg_stanford':
                first_shuffled_pass = [[numpy.asarray(random.sample(v[0].tolist(), k=len(v[0]))), word_vectors[k], k] for k, v in sub_images_individuals.items()]
                second_shuffled_pass = [[numpy.asarray(random.sample(v[1].tolist(), k=len(v[1]))), word_vectors[k], k] for k, v in sub_images_individuals.items()]
            else:
                first_shuffled_pass = [[numpy.asarray(random.sample(v[i].tolist(), k=len(v[0]))), word_vectors[k], k] for k, v in sub_images_individuals.items() for i in range(7)]
                second_shuffled_pass = [[numpy.asarray(random.sample(v[7].tolist(), k=len(v[1]))), word_vectors[k], k] for k, v in sub_images_individuals.items()]

            # Images for the 'averaged' condition

            average_pass = [[numpy.average(v, axis=0), word_vectors[k], k] for k, v in sub_images_individuals.items()]

            # Images for the 'randomized' condition

            random_indices  = random.sample([k for k in range(len(second_pass))], k=len(second_pass))
            good_keys = [k for k in sub_images_individuals.keys()]
            good_images = [v for k, v in sub_images_individuals.items()]
            randomized_keys = [good_keys[i] for i in random_indices]
            randomized_images_individuals = {k : v for k, v in zip(randomized_keys, good_images)}
            if args.dataset != 'eeg_stanford':
                first_randomized_pass = [[v[0], word_vectors[k], k] for k, v in randomized_images_individuals.items()]
                second_randomized_pass = [[v[1], word_vectors[k], k] for k, v in randomized_images_individuals.items()]
            else:
                first_randomized_pass = [[v[i], word_vectors[k], k] for k, v in randomized_images_individuals.items() for i in range(7)]
                second_randomized_pass = [[v[7], word_vectors[k], k] for k, v in randomized_images_individuals.items()]

            # Images for the 'men' condition

            if args.dataset != 'eeg_stanford':
                first_men_pass = [[v[0], men_vectors[good_keys.index(k)], k] for k, v in sub_images_individuals.items()]
                second_men_pass = [[v[1], men_vectors[good_keys.index(k)], k] for k, v in sub_images_individuals.items()]
            else:
                first_men_pass = [[v[i], word_vectors[k], k] for k, v in sub_images_individuals.items() for i in range(7)]
                second_men_pass = [[v[7], word_vectors[k], k] for k, v in sub_images_individuals.items()]

            if args.evaluation_method == 'ranking':

                logging.info('Now starting the ranking accuracy evaluation...')
                for test_index, test_instance in tqdm(enumerate(second_pass)):

                    train_dict, test_dict = train_test_split(args, first_pass, second_pass, first_shuffled_pass, second_shuffled_pass, first_randomized_pass, second_randomized_pass, average_pass, first_men_pass, second_men_pass, [int(test_index)], test_instance)

                    for test_key, train_data in train_dict.items():
                         
                        if not args.blind_test or test_key != 'original':
                            test_data = test_dict[test_key][0]
                            test_id = test_dict[test_key][1]
                            
                            # NOT_SURE_MARKER

                            predictions = get_predictions(args, model, train_data, test_data)

                            if args.dataset != 'eeg_stanford':
                                sims = {k : cos(predictions, word_vectors[k]) for k, v in sub_images_individuals.items() if len(v) == 2}  
                            else: 
                                sims = {k : cos(predictions, word_vectors[k]) for k, v in sub_images_individuals.items()}  
                            rankings = {k : sims[k] for k in sorted(sims, key = sims.get, reverse = True)}
                            rank = 0
                            for k, sim in rankings.items():
                                rank += 1
                                if k == test_id:
                                    break
                            if args.dataset != 'eeg_stanford':
                                accuracy = 1 - ((rank - 1) / (len([k for k, v in sub_images_individuals.items() if len(v) == 2]) - 1))
                            else:
                                accuracy = 1 - ((rank - 1) / (len([k for k, v in sub_images_individuals.items()]) - 1))

                        elif args.blind_test and test_key == 'original':
                            accuracies = []
                            for blind_test_instance in test_dict[test_key]:

                                test_data = blind_test_instance[0]
                                test_id = blind_test_instance[1]

                                predictions = get_predictions(args, model, train_data, test_data)

                                if args.dataset != 'eeg_stanford':
                                    sims = {k : cos(predictions, word_vectors[k]) for k, v in sub_images_individuals.items() if len(v) == 2}  
                                else: 
                                    sims = {k : cos(predictions, word_vectors[k]) for k, v in sub_images_individuals.items()}  
                                rankings = {k : sims[k] for k in sorted(sims, key = sims.get, reverse = True)}
                                rank = 0
                                for k, sim in rankings.items():
                                    rank += 1
                                    if k == test_id:
                                        break
                                if args.dataset != 'eeg_stanford':
                                    accuracy = 1 - ((rank - 1) / (len([k for k, v in sub_images_individuals.items() if len(v) == 2]) - 1))
                                else:
                                    accuracy = 1 - ((rank - 1) / (len([k for k, v in sub_images_individuals.items()]) - 1))
                                accuracies.append(accuracy)
                            accuracy = numpy.average(accuracies)

                        people_results[test_id].append(accuracy)
                        results[test_key].append(accuracy)
                        histogram_results[test_key].append(accuracy)

            ### Voxel analysis - run only on the ranking because it's faster

            #if args.voxel_analysis:
                #voxel_analysis(args, selected_voxels_collection, masker, aal_atlas, talairach_atlas, model_path)

            ### Pairwise evaluation
            if args.evaluation_method == 'pairwise':

                logging.info('Now moving on to the pairwise evaluation (it\'s gonna be slow...)')
                combinations = itertools.combinations(range(len(first_pass)), 2)

                for test_combination in tqdm(combinations):
                    train_dict, test_dict = train_test_split(first_pass, second_pass, test_combination)

                    for test_key, train_data in train_dict.items():

                        test_data = [test_dict[test_key][i][0] for i in (0, 1)]
                        test_nlp_vectors = [test_dict[test_key][i][1] for i in (0, 1)]
                        test_id = [test_dict[test_key][i][2] for i in (0, 1)]
                        
                        predictions = get_predictions(args, model, train_data, test_data)

                        #m.fit([item[0] for item in train_data], [item[1] for item in train_data])

                        #predictions = m.predict([item[0] for item in test_data])
                        wrong_prediction = cos(predictions[0], test_nlp_vectors[1]) + cos(predictions[1], test_nlp_vectors[0])
                        good_prediction = cos(predictions[0], test_nlp_vectors[0]) + cos(predictions[1], test_nlp_vectors[1])

                        if good_prediction > wrong_prediction:
                            results[test_key].append(float(1))
                            histogram_results[test_key].append(1)
                        else:
                            results[test_key].append(float(0))
                            histogram_results[test_key].append(0)

            print('\tSubject {}'.format(s))
            print_results(args, s, model_path, results, final_results)

        elif 1 == 2:

            # Classification of brain images category
            category_decoding()


# Printing out the average and median results

logging.info('Final results for the {} model'.format(args.model))
for test_index, test_results in final_results.items():
    print('\tAverage result across subject using the {} vectors: {}'.format(test_index, numpy.average(test_results)))


if args.write_to_file:
 
    # Writing to text files
    
    test_path = os.path.join('plots', args.dataset, 'average_results', args.model)
    if args.model == 'rsa':
        test_path = os.path.join('plots', args.dataset, 'average_results', model_name, args.distance_measure)

    os.makedirs(test_path, exist_ok = True)

    # Average model results for plotting later on
    # model, vectors, evaluation method, amount of voxels, test index, average, std
    if not args.find_stable_voxels:
        args.amount_stable_voxels = 68049
    
    results_path = os.path.join('results')
    os.makedirs(results_path, exist_ok = True)
    with open(os.path.join(results_path, 'final_results_all_models.txt'.format(args.evaluation_method)), 'a') as final_out:
        #final_out.write('Results for model {} - {} test:\n\n'.format(model_name, args.evaluation_method))
        for test_index, test_results in final_results.items():
            final_out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.model, args.vectors, args.evaluation_method, args.amount_stable_voxels, test_index, numpy.average(test_results), numpy.std(test_results)))
            #final_out.write('Average and median result across subject using the {} vectors:\t{}, {}\n\n'.format(test_index, numpy.average(test_results), numpy.median(test_results)))
            


    #Plotting an histogram with the comparisons of the results

    plt.hist([test_results for test_id, test_results in histogram_results.items()], label = [test_id for test_id, test_results in histogram_results.items()], alpha = 0.5)
    plt.legend()

    plt.title = ('Histogram for the {} model - {} test'.format(model_name.capitalize(), args.evaluation_method))
    plt.savefig(os.path.join(test_path, '{}_{}_voxels_histogram.png'.format(args.model, args.evaluation_method, args.amount_stable_voxels))) 


    if args.dataset == 'mmms_fame' and args.model == 'ridge':

        # Per-person accuracy scores    

        people_medians = {person_key : numpy.median(l) for person_key, l in people_results.items()}
        people_results_ordered = {person_key : people_results[person_key] for person_key in sorted(people_medians, key = people_medians.get, reverse = True)}
        with open(os.path.join(test_path, '{}_people_results.txt'.format(args.evaluation_method)), 'a') as people_out:
            people_out.write('Median accuracy scores per id/person\n\n')
            for person_key, median_score in people_results_ordered.items():
                people_out.write('{}\t{}\n'.format(person_key, median_score))

