import numpy
import os

from matplotlib import pyplot
from scipy import stats

from io_utils import prepare_folder
from feature_selection.feature_selection import feature_ranking_folder 
from plot_scripts.plot_violins import plot_violins

def plot_decoding_scores_comparison(args):

    #subjects = final_subjects(args)
    subjects = list(range(1, 33+1))
    labels = list()
    accs = list()

    models = [\
              'BERT_large_en_mentions',\
              'BERT_base_en_mentions', \
              #'BERT_base_it_mentions', \
              #'ELMO_small_en_mentions',\
              'ELMO_original_en_mentions',\
              'LUKE_base_en_mentions', 
              'LUKE_large_en_mentions',\
              'w2v', \
              #'it_w2v', \
              'transe', \
              'wikipedia2vec', \
              #'it_wikipedia2vec', \
              #'mBERT', \
              ]

    for comp_model in models:
        args.word_vectors = comp_model
        folder = prepare_folder(args)

        #for f_selec in os.listdir(folder):
            
        file_name = '{}_{}_results.txt'.format(args.word_vectors, \
                         args.evaluation_method)
        file_path = os.path.join(folder, 'no_reduction', file_name)

        if not os.path.exists(file_path):
            print('missing: {}'.format(file_path))
        else:
            labels.append(comp_model.replace('_en_mentions', ''\
                          ).replace('_', ' '))
            with open(file_path) as i:
                lines = [l.strip().split('\t')[0] for l in i.readlines()]
            if args.experiment_id == 'two':
                lines = lines[1:]
            ### Final subjects are from 1 to 33; since the first line in txt is just the 
            ### line corresponding to the header we can take the lines using final_subjects directly
            lines = [lines[i] for i in subjects]
            
            assert len(lines) == len(subjects)
            accs.append(lines)

    random_baseline = .5
    ### Printing out what's missing
    if len(accs) == 0:
        print('missing {}'.format(file_path))
    ### If there's at least one result, proceed
    else:

        accs = numpy.array(accs, dtype=numpy.double) 

        #plot_path = os.path.join('plots', args.experiment_id, \
        #                         '{}_scores_comparison'.format(args.analysis), \
        #                         args.subsample, args.entities, \
        #                         args.semantic_category)
        plot_path = prepare_folder(args).replace(args.word_vectors, '')\
                                        .replace(args.analysis, '{}/models_comparison'.format(args.analysis))\
                                        .replace('results', 'plots')
        plot_path = os.path.join(plot_path, args.evaluation_method)
        os.makedirs(plot_path, exist_ok=True)
        #plot_path  = os.path.join(plot_path, 'decoding_{}_{}.pdf'.format(\
        plot_path  = os.path.join(plot_path, '{}_{}_{}.jpg'.format(args.analysis,\
                                  args.entities, args.semantic_category))

        plot_title = 'Comparing individual-level {} scores \n'\
                        '{} - {} - N={}'.format(args.analysis, \
                        args.entities, args.semantic_category, \
                        accs.shape[1]).replace('_', ' ')

        plot_violins(accs, labels, plot_path, plot_title, random_baseline)
