import os
import sklearn
import numpy
import random
import argparse

from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from utils.load_eeg import load_eeg_vectors

parser = argparse.ArgumentParser()
parser.add_argument('--categories', action='store_true', default=False)
args = parser.parse_args()

final_scores = []

for s in range(1, 11):
    sub_scores = []
    print('Now starting subject {}'.format(s))
    eeg_vectors = load_eeg_vectors(s, categories=args.categories)
         
    classes = {k : i+1 for i, k in enumerate(eeg_vectors.keys())}
    ordered_dataset = [(vec, classes[k]) for k, v in eeg_vectors.items() for vec in v]
    randomized_dataset = random.sample(ordered_dataset, k=len(ordered_dataset))

    for p in [3300]:
        pca = PCA(n_components=p)
        print('Now reducing the dataset dimensionality to {}...'.format(p))
        reduced_dataset = pca.fit_transform([k[0] for k in randomized_dataset])
        dataset = [(one, two[1]) for one, two in zip(reduced_dataset, randomized_dataset)]

        steps = int(len(dataset)/12)
        beginning = 0
        for it in range(12):

            train_data = dataset[:beginning] + dataset[beginning+steps:]
            train_samples = [i[0] for i in train_data]
            train_targets = [i[1] for i in train_data]
            #print('Train data length: {}'.format(len(train_samples)))

            test_data = dataset[beginning:beginning+steps]
            test_samples = [i[0] for i in test_data]
            test_targets = [i[1] for i in test_data]
            #print('Test data length: {}'.format(len(test_samples)))

            classifier = OneVsRestClassifier(SVC()).fit(train_samples, train_targets)
            score = classifier.score(test_samples, test_targets)
            sub_scores.append(score)
            
            beginning += steps

        print('score for subject {}: {}'.format(s, numpy.average(sub_scores)))
    final_scores.append(sub_scores)

import pdb; pdb.set_trace()
