import matplotlib
import numpy
import os

from matplotlib import pyplot
from scipy import stats

from io_utils import prepare_folder
from feature_selection.feature_selection import feature_ranking_folder 
from plot_scripts.plot_violins import plot_violins

def plot_decoding_results_breakdown(args):

    matplotlib.rc('image', cmap='Dark2')

    subjects = 33
    labels = list()
    accs = dict()

    folder = prepare_folder(args)
    for f_selec in os.listdir(folder):
        
        file_name = '{}_{}_results.txt'.format(args.word_vectors, \
                     args.evaluation_method)
        file_path = os.path.join(folder, f_selec, file_name)

        if not os.path.exists(file_path):
            print('missing: {}'.format(file_path))
        else:
            with open(file_path) as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            #if args.experiment_id == 'two':
            #    lines = lines[1:]
            header_indices = [l for l in enumerate(lines[0]) if 'Accuracy' in l[1] or \
                                                             'mixed_place_place' in l[1] or\
                                                             'mixed_person_place' in l[1] or\
                                                             'entity_persona' in l[1] or \
                                                             'entity_person_person' in l[1] or\
                                                             'entity_luogo' in l[1] or \
                                                             'entity_place_place' in l[1] or\
                                                             'entity_person_place' in l[1] or\
                                                             'category_persona' in l[1] or \
                                                             'category_person_person' in l[1] or\
                                                             'category_luogo' in l[1] or \
                                                             'category_place_place' in l[1] or\
                                                             'category_person_place' in l[1] or\
                                                             'mixed_persona' in l[1] or \
                                                             'mixed_person_person' in l[1] or \
                                                             'person_place' in l[1] or \
                                                             'famous_famous' in l[1] or \
                                                             'person_place_famous_famous' in l[1] or \
                                                             'place_place' in l[1] or \
                                                             'place_place_famous_famous' in l[1] or \
                                                             'person_person' in l[1] or \
                                                             'person_person_famous_famous' in l[1] or \

                                                             'mixed_luogo' in l[1]]


            ### Final subjects are from 1 to 33; since the first line in txt is just the 
            ### line corresponding to the header we can take the lines using final_subjects directly
            assert len(lines) == subjects+1
            lines = [lines[i] for i in range(1, subjects+1)]
            for h_i, h in header_indices:
                labels.append(f_selec)
                if h not in accs.keys():
                    accs[h] = list()
                for l in lines:
                    accs[h].append(float(l[h_i]))


    ### Printing out what's missing
    if len(accs.keys()) == 0:
        print('missing {}'.format(file_path))
    
    ### If there's at least one result, proceed
    else:

        random_baseline = 0.5
        assert len(list(set([len(v) for k, v in accs.items()]))) == 1
        #accs = numpy.array(accs, dtype=numpy.double) 

        comp_model = args.word_vectors.replace('_en_mentions', '')
        #plot_path = os.path.join('plots', args.experiment_id, \
        #                         '{}_results_breakdown'.format(args.analysis), \
        #                         args.subsample, args.entities, \
        #                         args.semantic_category)
        plot_path = prepare_folder(args).replace(args.analysis, \
                                                 '{}/results_breakdown'.format(args.analysis))\
                                        .replace('results', 'plots')
        plot_path = os.path.join(plot_path, args.evaluation_method)
        os.makedirs(plot_path, exist_ok=True)
        ### txt file to compute correlations
        text_path = os.path.join(plot_path, \
                    '{}_{}_{}_{}_breakdown.txt'.format(\
                    comp_model, args.entities, args.semantic_category, \
                    args.analysis))
        with open(text_path,'w', encoding='utf-8') as o:
            for k in accs.keys():
                o.write('{}\t'.format(k))
            o.write('\n')
            for k_i in range(len(accs['Accuracy'])):
                for k, v in accs.items():
                    o.write('{}\t'.format(v[k_i]))
                o.write('\n')

        plot_path = os.path.join(plot_path, \
                    #'{}_{}_{}_decoding_breakdown.pdf'.format(\
                    '{}_{}_{}_{}_breakdown_average{}.jpg'.format(\
                    comp_model, args.entities, args.semantic_category, \
                    args.analysis, args.average))

        ### Main plot properties

        plot_title = 'Comparing {} scores for various \n'\
                        'data splits for {} \n'\
                        '- {} - {} - N={}'.format(\
                        args.analysis, \
                        comp_model, \
                        args.entities, \
                        args.semantic_category, \
                        len(accs['Accuracy'])).replace('_', ' ')
        plot_title = plot_title.replace('_', ' ')
        plot_violins(accs, labels, plot_path, plot_title, random_baseline)
