import argparse
import numpy
import os
import re

from matplotlib import pyplot
from tqdm import tqdm

from io_utils import ExperimentInfo, LoadEEG

choices = ['aggregate', 'stability', 'noisiness', 'distinctiveness']

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', \
                    type=str, \
                    required=True, help='Indicates where \
                    the experiment files are stored')

args = parser.parse_args()
args.experiment = 'one'
args.average = 0
args.individuals_only = False
args.analysis = 'decoding'
args.time_resolved = False
args.restrict_words = 'people_and_places'
args.subsample = True

experiment = ExperimentInfo(experiment=args.experiment, \
                            data_folder=args.data_folder)

n_subjects = experiment.subjects
data_paths = experiment.eeg_paths
base_folder = 'feature_visualization'

for feat in choices:
    print(feat)
    folder = os.path.join(base_folder, feat)
    os.makedirs(folder, exist_ok=True)
    
    for n in tqdm(range(n_subjects)):

        data_path = data_paths[n]
        eeg_data = LoadEEG(args, data_path, n)
        eeg = eeg_data.data_dict
        times = eeg_data.times
        shape = eeg[list(eeg.keys())[0]].shape

        ### Feature selection
        entities_marker = 'individuals_only' if args.individuals_only else 'individuals_and_categories'
        file_path = os.path.join('feature_ranking', \
                                'sub-{:02}'.format(n+1), \
                                args.restrict_words, \
                                entities_marker, \
                                '{}.scores'.format(feat))

        with open(file_path) as i:
            lines = [l.strip().split('\t') for l in i.readlines()][1:]

        features_per_comb = {(l[0], l[1]) : l[2:] for l in lines}
        features_per_comb = {k : [w.split(',') for w in v] for k, v in features_per_comb.items()}
        features_per_comb = {k : {float(val[0]) : float(val[1]) for val in v} for k, v in features_per_comb.items()}

        all_selections = dict()
        for k, v in features_per_comb.items():
            for d, val in v.items():
                if d not in all_selections.keys():
                    all_selections[d] = [val]
                else:
                    all_selections[d].append(val)

        max_dim = int(max(list(all_selections.keys())))
        features = list()
        for i in range(max_dim+1):
            value = numpy.median(all_selections[i])
            features.append(value)
        features = numpy.array(features).reshape(shape)
        fig, ax = pyplot.subplots(figsize=(12, 6))

        mat = ax.imshow(features)
        ax.set_xticks([i+.5 for i in range(len(times))])
        ax.set_xticklabels(times, fontsize='xx-small')
        pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
        pyplot.colorbar(mat, ax=ax)
        pyplot.savefig(os.path.join(folder, 'sub-{:02}_features.png'.format(n+1)), \
                        dpi=1200)
