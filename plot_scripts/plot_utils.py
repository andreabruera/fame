import mne
import matplotlib
import numpy
import os

from matplotlib import pyplot
from scipy import stats

def fdr(scores, times):

    ### Computing p-values
    relevant_times = [i for i, t in enumerate(times) if t>-.1 and t<=1.]

    p_values = list()
    for t in relevant_times:
        t_scores = scores[:, t]
        p = stats.ttest_1samp(t_scores, popmean=.5, alternative='greater')[1]
        p_values.append(p)

    _, p_corrected = mne.stats.fdr_correction(p_values)
    corrected_p_times = [relevant_times[i] for i, p in enumerate(p_corrected) if p<=.05]

    return corrected_p_times

def plot_time_resolved_classification(args, scores, times, corrected_p_times):

    individuals_only = '_individuals_only' if args.individuals_only else '_all_entities'
    base_folder = 'classification_plots'
    os.makedirs(base_folder, exist_ok=True)

    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(10., 5.))

    average = numpy.nanmean(scores, axis=0)
    ax.plot(times, average)
    for vec in scores:
        ax.scatter(x=times, y=vec, alpha=0.1)
    ax.hlines(y=0.5, xmin=times[0], xmax=times[-1], \
              linestyles='dashed', linewidths=.1, \
              colors='darkgray')
    ax.set_title('Decoding accuracies for {}{}\n'\
                 'averaging {} ERPs'.format(\
                 args.analysis.replace('_', ' '), \
                 individuals_only.replace('_', ' '), \
                 args.average))

    significant_xs = [times[t] for t in corrected_p_times]
    significant_ys = [average[t] for t in corrected_p_times]
    if len(significant_xs) >= 1:
        ax.scatter(x=significant_xs, y=significant_ys, \
                   edgecolors='black', color='white', \
                   linewidths=1., label='p<=0.05')
    ax.legend()
    
    pyplot.savefig(os.path.join(base_folder, '{}_average_{}{}.png'.format(args.analysis, args.average, individuals_only)), dpi=600) 
    pyplot.clf()
