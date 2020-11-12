import numpy
import os
import re
import collections
import pickle
import argparse
import random
import sklearn
import itertools

from tqdm import tqdm
from sklearn.linear_model import Ridge
from utils.evaluation import predict_ranking, predict_pairwise
from utils.brain_images import find_stable_voxels, pca
from utils.extract_word_lists import Entities
from utils.load_eeg import load_eeg_vectors 

parser = argparse.ArgumentParser()
parser.add_argument('--words', choices=['wakeman_henson', 'eeg_stanford', 'mitchell'], required=True, help='Indicates which words to extract')
parser.add_argument('--modality', choices=['eeg', 'fmri'], default='eeg', help='Indicates which brain data modality to use')
parser.add_argument('--analysis', choices=['decoding', 'encoding'], default='encoding', help='Indicates whether to encode or decode')
parser.add_argument('--shots', choices=['zero', 'one'], default='one', help='Indicates which testing setup to use')
parser.add_argument('--evaluation_method', choices=['ranking', 'pairwise'], default='ranking', help='Indicates which evaluation method to use')
parser.add_argument('--feature_selection', choices=['pca', 'stability', 'none'], default='pca', help='Indicates how to reduce the dimensionality of the brain data. Choose among \'pca\', \'stability\' and \'no\'')
parser.add_argument('--random_assignment', action='store_true', default=False, help='Indicates whether to randomly map brain and word vectors')
parser.add_argument('--shuffled_vectors', action='store_true', default=False, help='Indicates whether to randomly shuffle brain vectors')
parser.add_argument('--smoothed_fmri', action='store_true', default=False, help='Indicates whether to use smoothed or non-smoothed fmri images')
parser.add_argument('--categories', action='store_true', default=False, help='Indicates whether to use as words individuals or categories')
args = parser.parse_args()
print(vars(args))

ridge_model = Ridge(alpha=1.0)
'''
cuda = torch.device('cuda:{}'.format(args.cuda_device))
torch.manual_seed(0)
torch.cuda.manual_seed(0)
loss_function = torch.nn.CosineEmbeddingLoss()

torch_perceptron = torch_linear(output_dimensions, input_dimensions).float()
torch_perceptron = torch_perceptron.to(device=cuda)
torch_perceptron.zero_grad()
model = [torch_perceptron, loss_function, cuda]
'''

### Loading word lists and word_vectors

ents = Entities(args.words)
ents_and_cats = ents.words


### Balancing the Wakeman & Henson dataset

if args.words == 'wakeman_henson':
    people_counter = collections.defaultdict(int)
    for k, v in ents_and_cats.items():
        people_counter[v] += 1

    threshold = min([v for k, v in people_counter.items() if v > 10])
    balanced_cats = [k for k, v in people_counter.items() if v > 10]
    people_selection = collections.defaultdict(list)
    for k, v in ents_and_cats.items():
        if v in balanced_cats:
            people_selection[v].append(k)
    people_selection = {k : random.sample(v, k=len(v))[:threshold] for k, v in people_selection.items()}
    ents_and_cats = {n : k for k, v in people_selection.items() for n in v}

### For categories, substituting the entity-level word vector with the category-level word vector
if args.categories:
    ent_list = [w for w, v in ents_and_cats.items() if v != '']
    category_vectors = ents.category_vectors
    word_vectors = {ents_and_cats[e] : category_vectors[ents_and_cats[e]] for e in ent_list}
else:
    ent_list = [w for w in ents_and_cats.keys() if re.sub('[0-9]', '', str(w)) != '']
    word_vectors = ents.word_vectors

### Preparing the dictionary for collecting the results
sub_accuracies = collections.defaultdict(list) 

### Setting the number of subjects
if args.words == 'wakeman_henson':
    subjects = 17
elif args.words == 'eeg_stanford':
    #subjects = 11
    subjects = 3
elif args.words == 'mitchell':
    subjects = 10

for sub in range(1, subjects):

    print('Subject {}'.format(sub))

    ### Loading brain images, mostly from pickle
    if args.words == 'wakeman_henson':
        if args.modality == 'eeg':
            with open('resources/wakeman_henson_stimuli.txt') as input_file:
                lines = [l.strip().split('\t') for l in input_file.readlines()]
            names_to_ids = {re.sub('\..+', '', l[0]) : l[1] for l in lines if len(l) > 2}
            original_images_individuals = pickle.load(open('/import/cogsci/andrea/dataset/eeg_images_new/sub-{:02}_eeg_vecs.pkl'.format(sub), 'rb'))
            sub_images_individuals = {names_to_ids[k] : v for k, v in original_images_individuals.items() if k in names_to_ids.keys()}
        elif args.modality == 'fmri':
            if args.smoothed_fmri:
                sub_images_individuals = pickle.load(open('/import/cogsci/andrea/github/fame/data/wakeman_henson_updated_pickles/fmri_sub_{:02}_smoothed.pkl'.format(sub), 'rb'))
            else:
                sub_images_individuals = pickle.load(open('/import/cogsci/andrea/github/fame/data/wakeman_henson_updated_pickles/fmri_sub_{:02}.pkl'.format(sub), 'rb'))
    elif args.words == 'mitchell':
        sub_images_individuals = pickle.load(open('/import/cogsci/andrea/dataset/mitchell_pickles/sub_{:02}.pkl'.format(sub), 'rb'))
    elif args.words == 'eeg_stanford':
        sub_images_individuals = load_eeg_vectors(sub, ent_list)

    ### Updating the list of words according to the actual vector availability

    ent_list = [w for w in ent_list if w in sub_images_individuals.keys()]

    ### Selecting the most relevant features for the brain images, if required
    if args.feature_selection == 'stability':
        sub_images_individuals = find_stable_voxels(500, sub_images_individuals)[0]
    elif args.feature_selection == 'pca':
        sub_images_individuals = pca(sub_images_individuals)

    if args.analysis == 'encoding':
        ### Using word vectors as input data and brain images as target data
        input_data = word_vectors.copy()

        ### Reducing all brain images to only one average brain image per individual
        target_data = {k : numpy.average(v[:2], axis=0) for  k, v in sub_images_individuals.items()}

    else:
        ### Using brain images as input data, word vectors as target data
        input_data = sub_images_individuals.copy()
        

        ### Reducing all word vectors to only the first one
        #full_target_data = category_vectors if args.categories else word_vectors
        target_data = {k : v[0] for k, v in word_vectors.items()}

    ### Preparing evaluation samples
    samples = collections.defaultdict(list)
    if args.random_assignment:
        random_ents = random.sample(ent_list, k=len(ent_list))
        random_mapping = {r_ent : real_ent for r_ent, real_ent in zip(random_ents, ent_list)}
    for e in ent_list:
        if args.random_assignment:
            e = random_mapping[e]
        current_cat = ents_and_cats[e]
        target_class = current_cat if args.categories else e
        if args.shuffled_vectors:
            samples[target_class] += [(numpy.array(random.sample(vec.tolist(), k=len(vec.tolist())), dtype=numpy.single), target_data[target_class], target_class) for vec in input_data[e]]
        else:
            samples[target_class] += [(vec, target_data[target_class], target_class) for vec in input_data[e]]

    ### Zero-shot evaluation

    if args.shots == 'zero':

        for e, test_data in tqdm(samples.items()):
            training_data = [s for e_two, test_data_two in samples.items() for s in test_data_two if e_two != e]
            test_data = [(s[0], s[2]) for s in test_data]
            ridge_model.fit([i[0] for i in training_data], [i[1] for i in training_data])
            accuracies = []
            for s in test_data:
                accuracy = predict_ranking(ridge_model, s, target_data)
                accuracies.append(accuracy)
            sub_accuracies[sub].append(accuracies)

    ### One-shot evaluation
 
    elif args.shots == 'one':

        all_samples = [tpl for e, e_samples in samples.items() for tpl in e_samples]
        shuffled_samples = random.sample(all_samples, k=len(all_samples))
        fold_length = int(len(shuffled_samples)/10)
        starting_point = 0
        for i in tqdm(range(10)):

            accuracies = []
            ending_point = starting_point + fold_length
            training_data = shuffled_samples[:starting_point] + shuffled_samples[ending_point:]
            #print('Training length: {}\tTesting length: {}'.format(len(training_data), len(test_data)))
            ridge_model.fit([i[0] for i in training_data], [i[1] for i in training_data])
            test_data = [(k[0], k[2]) for k in shuffled_samples[starting_point:ending_point]]

            if args.evaluation_method == 'ranking':
                for s in test_data:
                    accuracy = predict_ranking(ridge_model, s, target_data)
                    accuracies.append(accuracy)
            if args.evaluation_method == 'pairwise':
                combs = itertools.combinations(test_data, 2)
                for comb in combs:
                    accuracies.append(predict_pairwise(ridge_model, comb, target_data))
            fold_score = numpy.nanmean(accuracies)
            #print('{}: \tstd: {}'.format(fold_score, numpy.nanstd(accuracies)))
            sub_accuracies[sub].append(fold_score)
            starting_point += fold_length

    ### Printing the within-subject results
    print(numpy.nanmean([numpy.nanmean(lst) for lst in sub_accuracies[sub]]))

### Printing the general results
print('Current setup: {}'.format(vars(args))) 
all_results = []
for sub, acc in sub_accuracies.items():
    current_acc = numpy.nanmean([numpy.nanmean(lst) for lst in acc])
    all_results.append(current_acc)
print('Mean accuracy: {}'.format(numpy.nanmean(all_results)))

### Pickling the results for further analyses
#with open('temp/encoding_results_{}.pkl'.format(args.words), 'wb') as o:
    #pickle.dump(sub_accuracies, o)
    
