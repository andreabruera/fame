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
plot_path = os.path.join('classification_plots', 'within_coarse')
os.makedirs(plot_path, exist_ok=True)


analyses = ['within_coarse_classification']
entity_types = ['individuals_only', 'individuals_and_categories']
subsamples = ['subsample']
coarses = ['luogo', 'persona']
#pcas = ['no_reduction', 'pca']
pcas = ['no_reduction']
#averages = [16,8,4]
averages = [0]
splits = [90]

for analysis in analyses:
    for average in averages:
        for subsample in subsamples:
            for coarse_i, coarse in enumerate(coarses):

                fig, ax = pyplot.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12,5))
                ### Plotting the lines
                ax[0].set_title('{}'.format(analysis.replace('_', ' ').capitalize()))

                folders = list()
                all_data = list()
                all_labels = list()

                for pca in pcas:
                
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
                                all_labels.append(', '.join(['average {}'.format(average), entity_type, pca]))
                        except AssertionError:
                            print('Couldn\'t find {}'.format(folder))

                assert len(all_labels) == len(folders)

                ### Preparing colors
                cmap = cm.get_cmap('nipy_spectral')
                colors = [cmap(i) for i in range(0, 256, int(256/len(folders)))]
                shift_counter = 0

                for f_i, f in enumerate(folders):


                    ### Recovering the labels for the output files
                    labels = all_labels[f_i]

                    number_of_files = int(len(os.listdir(f))/2)

                    
                    shift = shift_counter*0.001
                    shift_counter += 1
                    setup_data = list()

                    ### Reading the individual files
                    for i in range(1, number_of_files+1):
                        try:
                            with open(os.path.join(f, 'sub_{:02}_accuracy_{}_scores.txt').format(i, coarse)) \
                                    as input_file:
                                data = numpy.array([l.strip().split('\t') \
                                        for l in input_file.readlines()], dtype=numpy.double)

                            times = data[0, :]+shift
                            data = data[1, :]
                            setup_data.append(data)
                        except FileNotFoundError:
                            print('{}'.format(i))

                    assert len(setup_data) > 1
                    setup_data = numpy.array(setup_data)
                    '''
                    color = colors[f_i] if coarse_i == 0 \
                                        else colors[f_i+coarse_i+1]
                    '''

                    ### Checking for statistical significance
                    random_baseline = 0.25
                    ### T-test
                    '''
                    original_p_values = stats.ttest_1samp(setup_data, \
                                         popmean=random_baseline, \
                                         alternative='greater').pvalue
                    '''
                    ### Wilcoxon
                    significance_data = setup_data.T - random_baseline
                    original_p_values = list()
                    for t in significance_data:
                        p = stats.wilcoxon(t, alternative='greater')[1]
                        original_p_values.append(p)

                    assert len(original_p_values) == setup_data.shape[-1]

                    ### FDR correction
                    corrected_p_values = mne.stats.fdr_correction(original_p_values)[1]
                    significant_indices = [i for i, v in enumerate(corrected_p_values) if v<=0.05]

                    ax[0].errorbar(x=times, y=numpy.average(setup_data, axis=0), \
                                   yerr=stats.sem(setup_data, axis=0), \
                                   elinewidth=.5, linewidth=1.)
                    #ax[0].plot(times, numpy.average(setup_data, axis=0), color=color, linewidth=.5)
                    ax[0].scatter([times[t] for t in significant_indices], \
                               [numpy.average(setup_data, axis=0)[t] \
                                    for t in significant_indices], \
                                    color='white', edgecolors='black', \
                                    s=4.1, linewidth=.8)

                    ### Plotting the legend in a separate figure below
                    if coarse_i == 0:
                        x_text = .05
                        y_text = f_i+1
                    else:
                        x_text = .5
                        y_text = f_i+1
                    ax[1].scatter(x_text, y_text, label=f, alpha=1.)
                    ax[1].text(x_text, y_text, ' '.join((labels, coarse)))

                ### Removing all the parts surrounding the plot below
                ax[1].set_xlim(left=0., right=1.2)
                ax[1].spines['top'].set_visible(False)
                ax[1].spines['bottom'].set_visible(False)
                ax[1].spines['right'].set_visible(False)
                ax[1].spines['left'].set_visible(False)
                ax[1].get_xaxis().set_visible(False)
                ax[1].get_yaxis().set_visible(False)

                ax[0].hlines(y=random_baseline, xmin=times[0], xmax=times[-1], color='darkgrey', linestyle='dashed')
                #pyplot.show()
                #import pdb; pdb.set_trace()
                pyplot.savefig(os.path.join(plot_path, '{}_{}_{}_average_{}.png'.format(coarse, analysis, subsample, average)), dpi=600)
                pyplot.clf()
                pyplot.close(fig)
