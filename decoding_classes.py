import argparse
import collections
import numpy
import pickle
import random
import sklearn
import sklearn.multiclass
import sklearn.svm

from utils.brain_images import pca, find_stable_voxels
from utils.extract_word_lists import Entities
from utils.load_eeg import load_eeg_vectors

parser = argparse.ArgumentParser()
parser.add_argument('--entities', choices=['wakeman_henson', 'eeg_stanford', 'mitchell'], required=True, help='Indicates which words to extract')
parser.add_argument('--modality', choices=['eeg', 'fmri'], default='eeg', help='Indicates which brain data modality to use')
parser.add_argument('--shots', choices=['zero', 'one'], default='one', help='Indicates which testing setup to use')
parser.add_argument('--feature_selection', choices=['pca', 'stability', 'none'], default='pca', help='Indicates how to reduce the dimensionality of the brain data. Choose among \'pca\', \'stability\' and \'no\'')
parser.add_argument('--smoothed_fmri', action='store_true', default=False, help='Indicates whether to use smoothed or non-smoothed fmri images')
args = parser.parse_args()

ents = Entities(args.entities).words
sub_accuracies = collections.defaultdict(list)

### Setting the number of subjects
if args.entities == 'wakeman_henson':
    subjects = 17
elif args.entities == 'eeg_stanford':
    subjects = 11
elif args.entities == 'mitchell':
    subjects = 10

for sub in range(1, subjects):

    print('Subject {}'.format(sub))

    ### Loading brain images, mostly from pickle
    if args.entities == 'wakeman_henson':
        if args.modality == 'eeg':
            sub_images_individuals = pickle.load(open('/import/cogsci/andrea/dataset/eeg_images_new/sub-{:02}_eeg_vecs.pkl'.format(sub), 'rb'))
        elif args.modality == 'fmri':
            if args.smoothed_fmri:
                sub_images_individuals = pickle.load(open('/import/cogsci/andrea/github/fame/data/wakeman_henson_updated_pickles/fmri_sub_{:02}_smoothed.pkl'.format(sub), 'rb'))
            else:
                sub_images_individuals = pickle.load(open('/import/cogsci/andrea/github/fame/data/wakeman_henson_updated_pickles/fmri_sub_{:02}.pkl'.format(sub), 'rb'))
    elif args.entities == 'mitchell':
        sub_images_individuals = pickle.load(open('/import/cogsci/andrea/dataset/mitchell_pickles/sub_{:02}.pkl'.format(sub), 'rb'))
    elif args.entities == 'eeg_stanford':
        sub_images_individuals = load_eeg_vectors(sub, ents)

    ### Selecting the most relevant features, if required
    if args.feature_selection == 'stability':
        sub_images_individuals = find_stable_voxels(500, sub_images_individuals)[0]
    elif args.feature_selection == 'pca':
        sub_images_individuals = pca(sub_images_individuals)


    ### Preparing evaluation samples
    samples = collections.defaultdict(list)
    for e, cat in ents.items():
        if e in sub_images_individuals.keys():
            for vec in sub_images_individuals[e]:
                samples[cat].append((vec, e))

    for e in sub_images_individuals.keys():
        training_data = [(s[0], cat) for cat, tuples in samples.items() for s in tuples if s[1] != e]
        test_data = [(s[0], cat) for cat, tuples in samples.items() for s in tuples if s[1] == e]
        classifier = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC()).fit([i[0] for i in training_data], [i[1] for i in training_data])
        score = classifier.score([i[0] for i in test_data], [i[1] for i in test_data])
        sub_accuracies[sub].append(score)
    print(numpy.nanmean(sub_accuracies[sub]))

print(vars(args))
print(numpy.nanmean([numpy.nanmean(v) for k, v in sub_accuracies.items()]))
