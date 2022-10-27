import os
import matplotlib
import numpy
import pdb
import mne

from scipy import stats
from matplotlib import pyplot, cm

from io_utils import prepare_folder

class CurrentSetup:

    def __init__(self, analysis, entity_type, subsample, pca, average, split):
        self.analysis = analysis
        self.individuals_only = True if entity_type=='individuals_only' else False
        self.PCA = True if pca=='pca' else False
        self.subsample = True if subsample=='subsample' else False
        self.average = average
        self.training_split = split
        self.test_length=False
        self.feature_reduction='no_reduction'

plot_path = os.path.join('classification_plots', 'separate')
#plot_path = os.path.join('classification_plots', 'test', 'separate')
os.makedirs(plot_path, exist_ok=True)

analyses = ['fine_classification', 'coarse_classification']
#analyses = ['coarse_classification']
entity_types = ['individuals_only', 'individuals_and_categories']
#entity_types = ['individuals_and_categories']
#subsamples = ['subsample', 'all_time_points']
subsamples = ['subsample']
#pcas = ['no_reduction', 'pca']
pcas = ['no_reduction']
#averages = [16,8,4]
averages = [0]
splits = [90]
excluded_subjects = [18, 28, 31]

for analysis in analyses:
    for subsample in subsamples:

        fig, ax = pyplot.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12,5))

        folders = list()
        all_data = list()
        all_labels = list()

        for pca in pcas:
            for average in averages:
                for entity_type in entity_types:
                    args = CurrentSetup(analysis, entity_type, subsample, pca, average, split=90)
                    folder = prepare_folder(args)
                    try:
                        assert os.path.exists(folder)
                        number_of_files = len(os.listdir(folder))
                        if number_of_files <= 1: 
                            print('Less than one file in : {}'.format(folder))
                        else:
                            folders.append(folder)
                            all_labels.append(', '.join(['average {}'.format(average), entity_type]))
                            #all_labels.append(entity_type)
                    except AssertionError:
                        print('Couldn\'t find {}'.format(folder))

        assert len(all_labels) == len(folders)

        ### Preparing colors
        #cmap = cm.get_cmap('nipy_spectral')
        cmap = cm.get_cmap('plasma')
        #colors = [cmap(i) for i in range(64, , int(256/len(folders)))]
        colors = [cmap(i) for i in range(32, 220, int(220/len(folders)))]

        for f_i, f in enumerate(folders):

            plot_shift = f_i*0.001
            setup_data = list()

            ### Recovering the labels for the output files
            labels = all_labels[f_i]

            number_of_files = len(os.listdir(f))
            ### Reading the individual files
            for i in range(1, number_of_files+1):
                if i not in excluded_subjects:
                    try:
                        with open(os.path.join(f, 'sub_{:02}_accuracy_scores.txt').format(i)) \
                                as input_file:
                            data = numpy.array([l.strip().split('\t') \
                                    for l in input_file.readlines()], dtype=numpy.double)

                        times = data[0, :]+plot_shift
                        data = data[1, :]
                        setup_data.append(data)
                    except FileNotFoundError:
                        print('{}'.format(i))

            assert len(setup_data) > 1
            setup_data = numpy.array(setup_data)


            ### Plotting the lines
            ax[0].set_title('{}'.format(analysis.replace('_', '-grained ').capitalize()))
            ax[0].hlines(y=random_baseline, xmin=times[0], xmax=times[-1], color='darkgrey', linestyle='dashed')
            #ax[0].hlines(y=random_baseline, xmin=0, xmax=len(times), color='darkgrey', linestyle='dashed')
            #ax[0].plot(times, numpy.average(setup_data, axis=0), color=colors[f_i], linewidth=.5)
            ax[0].errorbar(x=times, y=numpy.average(setup_data, axis=0), \
                           yerr=stats.sem(setup_data, axis=0), \
                           #color=colors[f_i], \
                           elinewidth=.5, linewidth=1.)
                           #linewidth=.5)
            #for t_i, t in enumerate(times):
                #ax[0].violinplot(dataset=setup_data[:, t_i], positions=[t_i], showmedians=True)
            
            ax[0].scatter([times[t] for t in significant_indices], \
            #ax[0].scatter(significant_indices, \
                       [numpy.average(setup_data, axis=0)[t] \
                            for t in significant_indices], \
                            color='white', edgecolors='black', \
                            s=9., linewidth=.5)

            ### Plotting the legend in a separate figure below
            if f_i < (len(folders)/2):
                x_text = .05
                y_text = f_i+1
            else:
                x_text = 0.6
                y_text = f_i+1-int((len(folders)/2))
            ax[1].scatter(x_text, y_text, \
                          #color=colors[f_i], \
                          label=f, alpha=1.)
            ax[1].text(x_text+.05, y_text, labels.replace('_', ' '))

            ### Removing all the parts surrounding the plot below
            ax[1].set_xlim(left=0., right=1.2)
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['left'].set_visible(False)
            ax[1].get_xaxis().set_visible(False)
            ax[1].get_yaxis().set_visible(False)

        #pyplot.show()
        #import pdb; pdb.set_trace()
        pyplot.savefig(os.path.join(plot_path, '{}_{}.png'.format(analysis, subsample)), dpi=600)
        pyplot.clf()
        pyplot.close(fig)
