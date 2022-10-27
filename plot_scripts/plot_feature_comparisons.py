import os
import matplotlib
import numpy
import pdb
import mne
import argparse

from scipy import stats
from matplotlib import pyplot, cm

from io_utils import prepare_folder

import sys
sys.path.append('/import/cogsci/andrea/github/')

from grano.plot_utils import confusion_matrix_simple

class CurrentSetup:

    def __init__(self, entity_type, args):
        #self.analysis = 'decoding' 
        self.analysis = 'decoding' 
        #self.analysis = 'encoding' 
        #self.analysis = 'transfer_encoding' 
        self.individuals_only = True if entity_type=='individuals_only' else False
        self.feature_reduction = False
        self.subsample = True
        self.average = 0
        self.training_split = 90
        self.restrict_words = args.restrict_words
        #self.model = model
        '''
        self.evaluation_method = '_{}'.format(args.evaluation_method) if \
                                 args.evaluation_method == 'ranking' else ''
        '''
        self.evaluation_method = args.evaluation_method
        self.random = args.random
        self.time_resolved = args.time_resolved
        self.test_length = False

parser = argparse.ArgumentParser()
#parser.add_argument('--model', type=str, choices=['bert', ], required=True, \
                    #help='Indicates which model to plot')
parser.add_argument('--evaluation_method', type=str, \
                    choices=['pairwise', \
                             'ranking', \
                             ''], \
                    required=False, default='pairwise', \
                    help='Indicates which evaluation method to use')
parser.add_argument('--time_resolved', \
                    action='store_true', \
                    default=False, help='Runs the analyses on a time-by-time \
                    basis')
parser.add_argument('--random', \
                    action='store_true', \
                    default=False, help='Defines whether to randomize labels \
                    or not')
parser.add_argument('--restrict_words', \
                    choices=['people_and_places', 'people', 'places'], \
                    default='people_and_places', help='Restricts the analyses \
                    to a given category')
args = parser.parse_args()

plot_path = os.path.join('decoding_plots')
if isinstance(args.restrict_words, str):
    plot_path = os.path.join(plot_path, args.restrict_words)
os.makedirs(plot_path, exist_ok=True)

entity_types = ['individuals_only', 'individuals_and_categories']
#entity_types = ['individuals_and_categories']
#layers = list(range(1, 13)) if args.model=='bert' else ['']

excluded_subjects = [18, 28, 31]

for entity_type in entity_types:
    
    #fig, ax = pyplot.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12,5))
    fig, ax = pyplot.subplots(figsize=(12,8))

    plot_data = list()

    features = sorted(os.listdir(os.path.join('classification_per_subject', 'decoding', entity_type)))

    for feature in features:

        args = CurrentSetup(entity_type, args)
        args.feature_reduction = feature
        folder = prepare_folder(args)
        time_marker = 'time_resolved' if args.time_resolved else 'whole_epoch'
        random_marker = 'random' if args.random else 'true_evaluation'
        folder = os.path.join(folder, args.restrict_words, \
                                time_marker, random_marker)
        '''
        if isinstance(args.restrict_words, str):
            folder = os.path.join(folder, args.restrict_words)

        #for layer in layers:
        features = list(set([w.split('_')[0] for w in os.listdir(folder)]))
        features = [k for k in features if k!='people' and k!='places']
        features_list = features.copy()
        features = ['transe', 'w2v', 'wikipedia2vec', 'elmo', 'bert', 'ernie']
        assert set(features) == set(features_list)
        '''
        model = 'elmo'

        if model == 'bert' or model == 'ernie':
            file_name = '{}_full_sentence_layer_7_{}_results.txt'.format(model, args.evaluation_method)
        elif 'elmo' in model:
            file_name = '{}_unmasked_{}_results.txt'.format(model, args.evaluation_method)
        else:
            file_name = '{}_{}_results.txt'.format(model, args.evaluation_method)

        file_path = os.path.join(folder, file_name)
 
        if os.path.exists(file_path):

            with open(file_path) as i:
                 lines = numpy.array([l.strip().split('\t')[0] for l_i, l in enumerate(i.readlines()) \
                                     if l_i not in excluded_subjects][1:], dtype=numpy.single)
            plot_data.append(lines)
        else:
            print('missing path: {}'.format(file_path))

    ### Preparing colors
    cmap = cm.get_cmap('nipy_spectral')
    xs = list(range(1, len(plot_data)+1))
    colors = [cmap(i) for i in range(0, 256, int(256/(len(xs)+1)))]
        
    for feature_index, data in enumerate(plot_data):
        '''
        #layer = layer + 1 if args.model=='bert' else ''
        ax.bar(xs[feature_index]-.5, numpy.average(data), \
                  color=colors[feature_index], alpha=0.2, edgecolor='black', \
                  width=0.2)
        ax.scatter([xs[feature_index]-.5 for i in data], data, s=3.)
        '''
        ax.violinplot(data, \
                         positions=[feature_index+.5], \
                         showmedians=True)
    #color=colors[feature_index]) 

    ax.hlines(y=0.5, xmin=0., xmax=len(xs), \
                 linestyle='dashed', color='gray')
    ax.set_xticks([x-.5 for x in xs])
    ax.set_xticklabels(features)
    ax.set_title('Decoding for {}, using {} - '\
                    '{} evaluation'.format(\
                    args.restrict_words.replace('_', ' '), \
                    entity_type.replace('_', ' '), \
                    args.evaluation_method),\
                    pad=10)
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    '''

    ### Plotting the legend in a separate figure below
    ax[1].set_xlim(left=0., right=len(xs))
    ax[1].set_ylim(top=0., bottom=(len(plot_data)/2)+1.)

    for feature_index, data in enumerate(plot_data):

        #layer = layer + 1 if args.model=='bert' else ''
        if feature_index < len(plot_data)/2:
            x_text = .5
            y_text = feature_index+.5
        else:
            x_text = (len(xs)/2)+.5
            y_text = (feature_index-(len(plot_data)/2))+.5
        #x_text = .5
        #y_text = len(plot_data)-feature_index

        label='model {} - avg {} - median {}'.format(features[feature_index], \
                                             round(float(numpy.average(data)), 2), \
                                             round(float(numpy.median(data)), 2))

        ax[1].scatter(x_text-.1, y_text, alpha=1.)
        ax[1].text(x_text, y_text, label)

    ### Removing all the parts surrounding the plot below
    #ax[1].set_xlim(left=0., right=12.)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    '''

    out_path = os.path.join(plot_path, \
               'feature_comparison_{}_{}_{}_{}.png'.format(\
               args.restrict_words, args.analysis, \
               args.evaluation_method, entity_type))
    pyplot.savefig(out_path, dpi=300)
    pyplot.clf()

    ### Plotting the confusion matrix
    matrix_title = 'Correlation among per-subject model performances \n'\
                    'on the decoding task for {} - {} - '.format(\
                    args.evaluation_method, entity_type.replace('_', ' '),\
                    args.restrict_words)
    matrix = [[stats.pearsonr(l_one, l_two)[0] for l_two in plot_data] for l_one in plot_data]
    matrix_path = os.path.join(plot_path, \
               'feature_comparison_confusion_{}_{}_{}.png'.format(\
               args.analysis, args.evaluation_method, \
               entity_type))
    confusion_matrix_simple(matrix, features,matrix_title, matrix_path)
