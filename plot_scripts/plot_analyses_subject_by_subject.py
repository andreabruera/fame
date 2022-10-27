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
        self.feature_reduction = 'no_reduction'
        self.subsample = True if subsample=='subsample' else False
        self.average = average
        self.test_length = False
        self.training_split = split

plot_path = os.path.join('classification_plots', 'subject_by_subject')
os.makedirs(plot_path, exist_ok=True)

analyses = ['fine_classification', 'coarse_classification', 'transfer_classification']
analyses = ['coarse_classification']
entity_types = ['individuals_only', 'individuals_and_categories']
subsamples = ['subsample', 'all_time_points']
subsamples = ['subsample']
pcas = ['no_reduction', 'pca']
pcas = ['no_reduction']
averages = [16,8,4]
averages = [0]
splits = [90]

for analysis in analyses:
    for subsample in subsamples:


        folders = list()
        all_data = list()
        all_labels = list()

        for pca in pcas:
            for average in averages:
                for entity_type in entity_types:
                        args = CurrentSetup(analysis, entity_type, subsample, pca, average, split=90)
                        all_labels.append(', '.join(['average {}'.format(average), entity_type, pca]))
                        folder = prepare_folder(args)
                        try:
                            assert os.path.exists(folder)
                            number_of_files = len(os.listdir(folder))
                            if number_of_files <= 1: 
                                print('Folder missing conditions: {}'.format(folder))
                            else:
                                folders.append(folder)
                        except AssertionError:
                            print('couldn\'t find at all {}'.format(folder))

        ### Preparing colors
        cmap = cm.get_cmap('nipy_spectral')
        colors = [cmap(i) for i in range(0, 256, int(256/len(folders)))]

        for i in range(1, 34):

            fig, ax = pyplot.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12,5))
            for f_i, f in enumerate(folders):

                setup_data = list()

                labels = all_labels[f_i]

                number_of_files = len(os.listdir(f))
                try:
                    with open(os.path.join(f, 'sub_{:02}_accuracy_scores.txt').format(i)) as input_file:
                        data = numpy.array([l.strip().split('\t') for l in input_file.readlines()], dtype=numpy.double)

                    times = data[0, :]
                    data = data[1, :]
                    setup_data.append(data)
                except FileNotFoundError:
                    print('Could not find for subject {} file {}'.format(i, f))

                if len(setup_data) == 1:
                    setup_data = numpy.array(setup_data)

                    random_baseline = 0.125 if args.analysis == 'fine_classification' else 0.5

                    ax[0].set_title('Subject {} - {}'.format(i, analysis.replace('_', '-grained ').capitalize()))
                    ax[0].hlines(y=random_baseline, xmin=times[0], xmax=times[-1], color='darkgrey', linestyle='dashed')
                    ax[0].plot(times, numpy.average(setup_data, axis=0), color=colors[f_i], linewidth=.5)

                    if f_i < (len(folders)/2):
                        x_text = .5
                        y_text = f_i+1
                    else:
                        x_text = 6.5
                        y_text = f_i+1-int((len(folders)/2))
                    ax[1].scatter(x_text, y_text, color=colors[f_i], label=f, alpha=1.)
                    ax[1].text(x_text+.2, y_text-.2, labels)
                    #ax[1].frame_on=False
                    ax[1].set_xlim(left=0., right=12.)
                    ax[1].spines['top'].set_visible(False)
                    ax[1].spines['bottom'].set_visible(False)
                    ax[1].spines['right'].set_visible(False)
                    ax[1].spines['left'].set_visible(False)
                    ax[1].get_xaxis().set_visible(False)
                    ax[1].get_yaxis().set_visible(False)
                    #ax[1].legend()

            #pyplot.show()
            #import pdb; pdb.set_trace()
            pyplot.savefig(os.path.join(plot_path, 'sub-{:02}_{}_{}.png'.format(i, analysis, subsample)), dpi=600)
            pyplot.clf()
            pyplot.close()
