import numpy
import os

from matplotlib import pyplot
from scipy import stats

from io_utils import final_subjects, prepare_folder
from feature_selection.feature_selection import feature_ranking_folder 

def plot_feature_selection_comparison(args):

    subjects = final_subjects(args)
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
            labels.append(f_selec)
            with open(file_path) as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            #if args.experiment_id == 'two':
            #    lines = lines[1:]
            header_indices = [l for l enumerate(lines[0]) if 'Accuracy' in l or \
                                                             'entity' in l or \
                                                             'category' in l or \
                                                             'mixed' in l]
            ### Final subjects are from 1 to 33; since the first line in txt is just the 
            ### line corresponding to the header we can take the lines using final_subjects directly
            assert len(lines) == len(subjects)
            lines = [lines[i] for i in subjects]
            for h_i, h in header_indices:
                if h not in accs.keys():
                    accs[h] = list()
                for l in lines:
                    accs[h].append(l[h_i])

    ### Printing out what's missing
    if len(accs.keys()) == 0:
        print('missing {}'.format(file_path))
    ### If there's at least one result, proceed
    else:

        accs = numpy.array(accs, dtype=numpy.double) 

        plot_path = os.path.join('plots', args.experiment_id, 'decoding_results_breakdown', \
                                            args.subsample, args.entities, args.semantic_category)
        os.makedirs(plot_path, exist_ok=True)

        ### Preparing a double plot
        fig, ax = pyplot.subplots(nrows=2, ncols=1, \
                                  gridspec_kw={'height_ratios': [4, 1]}, \
                                  figsize=(12,5),constrained_layout=True)

        ### Main plot properties

        title = 'Comparing decoding scores for various \n'\
                        'feature selection methods for {} \n'\
                        '- {} - {} - N={}'.format(\
                        args.word_vectors, args.entities, \
                        args.semantic_category, accs.shape[1]).replace('_', ' ')
        title = title.replace('_', ' ')
        ax[0].set_title(title)

        random_baseline = .5
        ax[0].hlines(y=random_baseline, xmin=0, \
                     xmax=len(labels), color='darkgrey', \
                     linestyle='dashed')

        #ax[0].set_xticks([x for x in range(len(labels))])
        #ax[0].set_xticklabels(labels)

        ### Legend properties

        ### Removing all the parts surrounding the plot below
        ax[1].set_xlim(left=0., right=1.)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)

        ### Setting colors for plotting different setups
        #cmap = cm.get_cmap('plasma')
        #colors = [cmap(i) for i in range(32, 220, int(220/len(folders)))]

        number_lines = len(labels)

        line_counter = 0

        for feature_index, name_and_data in enumerate(accs.items()):
            name = name_and_data[0]
            data = name_and_data[1]
            ax[0].violinplot(data, positions=[feature_index], \
                                              showmedians=True)

            ### Plotting the legend in 
            ### a separate figure below
            line_counter += 1
            if line_counter <= number_lines/3:
                x_text = .1
                y_text = 0.+line_counter*.1
            elif line_counter > number_lines*0.33:
                x_text = .4
                y_text = 0.+line_counter*.066
            elif line_counter > number_lines*0.66:
                x_text = .7
                y_text = 0.+line_counter*.033

            label =  '{} - {} - avg: {} - median {} - p {}'.format(\
                                  name.replace('_', ' '), \
                                  labels[feature_index].replace('stability', ''), \
                                  round(numpy.average(data), 2), \
                                  round(numpy.median(data),2), \
                                  round(stats.wilcoxon(data, [.5 for y in data], \
                                      alternative='greater')[1], 5))
            ax[1].scatter(x_text, y_text, \
                          #color=colors[f_i], \
                          label=label, alpha=1.)
            ax[1].text(x_text+.05, y_text, label)


        pyplot.savefig(os.path.join(plot_path,\
                       '{}_decoding_results_breakdown_average{}.png'.format(args.word_vectors, args.average)),\
                       dpi=600)
        pyplot.clf()
        pyplot.close(fig)
