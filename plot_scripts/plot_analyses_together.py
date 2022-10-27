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

os.makedirs(os.path.join('classification_plots', 'together'), exist_ok=True)

analyses = ['fine_classification', 'coarse_classification']
entity_types = ['individuals_only', 'individuals_and_categories']
subsamples = ['subsample', 'all_time_points']
pcas = ['no_reduction', 'pca']
averages = [16,8,6,4]
splits = [90]

for average in averages:
    for subsample in subsamples:

        fig, ax = pyplot.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12,5))

        folders = list()
        all_data = list()
        all_labels = list()

        for analysis in analyses:
            for pca in pcas:
                for entity_type in entity_types:
                    args = CurrentSetup(analysis, entity_type, subsample, pca, average, split=90)
                    all_labels.append(', '.join([analysis, 'average {}'.format(average), entity_type, pca]))
                    folder = prepare_folder(args)
                    try:
                        assert os.path.exists(folder)
                        number_of_files = len(os.listdir(folder))
                        if number_of_files <= 1: 
                            print(folder)
                        else:
                            folders.append(folder)
                    except AssertionError:
                        print('couldn\'t find {}'.format(folder))

        ### Preparing colors
        cmap = cm.get_cmap('nipy_spectral')
        colors = [cmap(i) for i in range(0, 256, int(256/len(folders)))]

        for f_i, f in enumerate(folders):

            setup_data = list()

            labels = all_labels[f_i]

            number_of_files = len(os.listdir(f))
            for i in range(1, number_of_files+1):
                with open(os.path.join(f, 'sub_{:02}_accuracy_scores.txt').format(i)) as input_file:
                    data = numpy.array([l.strip().split('\t') for l in input_file.readlines()], dtype=numpy.double)

                times = data[0, :]
                data = data[1, :]
                setup_data.append(data)

            assert len(setup_data) > 1
            setup_data = numpy.array(setup_data)

            analysis = labels.split(',')[0]

            random_baseline = 0.5 if analysis == 'coarse_classification' else 0.125
            setup_data = setup_data - random_baseline

            original_p_values = stats.ttest_1samp(setup_data, popmean=random_baseline, alternative='greater').pvalue
            corrected_p_values = mne.stats.fdr_correction(original_p_values)[1]
            significant_indices = [i for i, v in enumerate(corrected_p_values) if v<=0.05]

            ax[0].set_title('{}'.format(analysis.replace('_', '-grained ').capitalize()))
            ax[0].hlines(y=0., xmin=times[0], xmax=times[-1], color='darkgrey', linestyle='dashed')
            ax[0].plot(times, numpy.average(setup_data, axis=0), color=colors[f_i], linewidth=.5)
            ax[0].scatter([times[t] for t in significant_indices], \
                       [numpy.average(setup_data, axis=0)[t] \
                            for t in significant_indices], \
                            color='white', edgecolors='black', \
                            s=9., linewidth=.5)
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

        pyplot.show()
        #pyplot.savefig(os.path.join('classification_plots', 'together', '{}_{}.png'.format(analysis, subsample)), dpi=600)
        pyplot.clf()
