import numpy
import os
import re

from matplotlib import pyplot
from scipy import stats

from io_utils import prepare_folder
from feature_selection.feature_selection import feature_ranking_folder 

def plot_feature_selection_comparison(args):

    #subjects = final_subjects(args)
    subjects = list(range(1, 33+1))
    labels = list()
    accs = list()

    if 'subsample' in args.subsample:
        folder = prepare_folder(args)
        base_folder = folder.split('subsample')[0]
        folders = [f for f in os.listdir(base_folder) if 'sub' in f]
        for fol in folders:
            for root, direc, filez in os.walk(os.path.join(base_folder, fol)):
                for f in filez:
                    print(f)
                    #if args.feature_selection_method in root or 'no_reduction' in root:
                    if 'no_reduction' in root:
                        if 'whole_trial' in args.analysis:
                            file_path = os.path.join(root, f)
                            label = root.split('/')[-1]
                            labels.append(label)
                            with open(file_path) as i:
                                lines = [l.strip().split('\t')[0] for l in i.readlines()]
                            if args.experiment_id == 'two':
                                lines = lines[1:]
                            ### Here there are 33 files, one for each subject 
                            accs.append(float(lines[-1]))
                        else:
                            if args.word_vectors in f and args.evaluation_method in f:
                            
                                #file_name = '{}_{}_results.txt'.format(args.word_vectors, \
                                #             args.evaluation_method)
                                #file_path = os.path.join(folder, f_selec, file_name)
                                file_path = os.path.join(root, f)
                                #print(file_path)

                                #if not os.path.exists(file_path):
                                #    print('missing: {}'.format(file_path))
                                #else:
                                label = fol.split('/')[-1]
                                labels.append(label)
                                with open(file_path) as i:
                                    lines = [l.strip().split('\t')[0] for l in i.readlines()]
                                if args.experiment_id == 'two':
                                    lines = lines[1:]
                                ### Final subjects are from 1 to 33; since the first line in txt is just the 
                                ### line corresponding to the header we can take the lines using final_subjects directly
                                lines = [lines[i] for i in subjects]
                                
                                assert len(lines) == len(subjects)
                                accs.append(lines)

    if 'subsample' not in args.subsample:
        folder = prepare_folder(args)
        #for f_selec in os.listdir(folder):
        for root, direc, filez in os.walk(folder):
            for f in filez:
                #print(f)
                if args.feature_selection_method in root or 'no_reduction' in root:
                    if 'whole_trial' in args.analysis:
                        file_path = os.path.join(root, f)
                        label = root.split('/')[-1]
                        labels.append(label)
                        with open(file_path) as i:
                            lines = [l.strip().split('\t')[0] for l in i.readlines()]
                        if args.experiment_id == 'two':
                            lines = lines[1:]
                        ### Here there are 33 files, one for each subject 
                        accs.append(float(lines[-1]))
                    else:
                        if args.word_vectors in f and args.evaluation_method in f:
                        
                            #file_name = '{}_{}_results.txt'.format(args.word_vectors, \
                            #             args.evaluation_method)
                            #file_path = os.path.join(folder, f_selec, file_name)
                            file_path = os.path.join(root, f)
                            #print(file_path)

                            #if not os.path.exists(file_path):
                            #    print('missing: {}'.format(file_path))
                            #else:
                            label = root.split('/')[-1]
                            labels.append(label)
                            with open(file_path) as i:
                                lines = [l.strip().split('\t')[0] for l in i.readlines()]
                            if args.experiment_id == 'two':
                                lines = lines[1:]
                            ### Final subjects are from 1 to 33; since the first line in txt is just the 
                            ### line corresponding to the header we can take the lines using final_subjects directly
                            lines = [lines[i] for i in subjects]
                            
                            assert len(lines) == len(subjects)
                            accs.append(lines)
    ### Reorganizing for the whole trial case
    if 'whole_trial' in args.analysis:

        collector = {k : list() for k in list(set(labels))}
        for l, a in zip(labels, accs):
            collector[l].append(a)
        accs = [v for k, v in collector.items()]
        labels = list(collector.keys())

    
    ### Printing out what's missing
    if len(accs) == 0:
        print('missing {}'.format(folder))
    ### If there's at least one result, proceed
    else:

        accs = numpy.array(accs, dtype=numpy.double) 

        #plot_path = os.path.join('plots', args.experiment_id, 'feature_selection_comparison', \
        #                                    args.subsample, args.entities, args.semantic_category)
        #plot_path = re.sub('encoding|decoding', 'feature_selection_comparison', \
        #                                        prepare_folder(args).replace('results', 'plots'))
        #plot_path = plot_path.replace('feature_sel', '{}_feature_sel'.format(args.analysis))
        plot_path = prepare_folder(args).replace('results', 'plots').replace(args.analysis, \
                                         '{}/feature_selection_comparison'.format(args.analysis))
        if 'whole_trial' not in args.analysis:
            plot_path = os.path.join(plot_path, args.evaluation_method)
        
        os.makedirs(plot_path, exist_ok=True)

        ### Preparing a double plot
        fig, ax = pyplot.subplots(\
                                  #nrows=2, ncols=1, \
                                  #gridspec_kw={'height_ratios': [4, 1]}, \
                                  figsize=(12,5), constrained_layout=True)

        ### Main plot properties

        title = 'Comparing {} scores for {} \n'\
                        'feature selection for {} \n'\
                        '- {} - {} - N={}'.format(args.analysis, \
                        args.feature_selection_method, args.word_vectors, args.entities, \
                        args.semantic_category, accs.shape[1]).replace('_', ' ')
        title = title.replace('_', ' ')
        ax.set_title(title)

        random_baseline = .5
        ax.hlines(y=random_baseline, xmin=0, \
                     xmax=len(labels), color='darkgrey', \
                     linestyle='dashed')

        #ax.set_xticks([x for x in range(len(labels))])
        #ax.set_xticklabels(labels)

        ### Legend properties

        '''
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
        '''

        number_lines = len(labels)

        line_counter = 0

        ### reorder by number

        #all_labels = [labels[feature_index] for feature_index in range(len(accs))]
        all_labels = [int(l.split('_')[-1]) if 'reduction' not in l else 150000 for l in labels]
        
        sorted_indices = [all_labels.index(name) for name in sorted(all_labels)]
        final_labels = list()

        #for feature_index, data in enumerate(accs):
        for feature_index, index in enumerate(sorted_indices):

            data = accs[index]
            label = all_labels[index]
            if label == 150000:
                label = 'No reduction'
            #ax.violinplot(data, positions=[feature_index], \
            ax.violinplot(data, positions=[feature_index], \
                                              showmedians=True)
            ''' 
            ## Plotting the legend in 
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

            label =  '{} - avg: {}'\
                     #'- median {} - p {}'\
                     ''.format(\
                                  #labels[feature_index].replace('stability', ''), \
                                  label, \
                                  round(numpy.average(data), 2), \
                                  #round(numpy.median(data),2), \
                                  #round(stats.wilcoxon(data, [.5 for y in data], \
                                  #    alternative='greater')[1], 5))
                                  )
            '''

            label = '{}\navg {} med {}'.format(label, round(numpy.average(data), 3), \
                                               round(numpy.median(data), 3))
            final_labels.append(label)

            '''
            ax[1].scatter(x_text, y_text, \
                          #color=colors[f_i], \
                          label=label, alpha=1.)
            ax[1].text(x_text+.05, y_text, label)
            '''

        ax.set_xticks(list(range(len(final_labels))))
        ax.set_xticklabels(final_labels, rotation=45, ha='right')

        pyplot.savefig(os.path.join(plot_path,\
                       '{}_{}_{}_feature_selection_comparison.png'.format(\
                       args.feature_selection_method, args.analysis, \
                       args.word_vectors)),\
                       dpi=600)
        pyplot.clf()
        pyplot.close(fig)
