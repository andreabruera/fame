import mne
import numpy
import os
import matplotlib
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats

def plot_violins(accs, labels, plot_path, title, random_baseline):

    ### Font setup
    # Using Helvetica as a font
    font_folder = '/import/cogsci/andrea/dataset/fonts/'
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

    ### Font size setup
    SMALL_SIZE = 23
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 27

    pyplot.rc('font', size=SMALL_SIZE)          # controls default text sizes
    pyplot.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    pyplot.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    pyplot.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    pyplot.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    pyplot.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ### Preparing a double plot
    if len(accs) < 10:
        fig, ax = pyplot.subplots(nrows=2, ncols=1, \
                                  gridspec_kw={'height_ratios': [7, 1]}, \
                                  figsize=(16,9),constrained_layout=True)
    else:
        fig, ax = pyplot.subplots(nrows=2, ncols=1, \
                                  gridspec_kw={'height_ratios': [7, 1]}, \
                                  figsize=(16,9),constrained_layout=True)

    ### Main plot properties

    #ax[0].set_title(title)
    ax[0].set_ylim(bottom=-.1, top=1.18)

    #ax[0].hlines(y=random_baseline, xmin=0, \
    ax[0].hlines(y=[0.1, 0.3, 0.7, 0.9], xmin=0, \
                 xmax=len(labels), color='black', \
                 linestyle=(0, (3, 10, 1, 10)), \
                 alpha=0.8, lw=0.5, zorder=1)
    ax[0].hlines(y=.5, xmin=0, \
                 xmax=len(labels), color='black', \
                 linestyle='dashdot', alpha=0.5, \
                 lw=0.5, zorder=1)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, \
                      #labelbottom=True, 
                      #labeltop=False, 
                      labelbottom=False, \
                      labeltop=True, \
                      )
    ax[0].tick_params(axis='y', which='both', left=False, right=False, \
                      #labelbottom=False, labeltop=False\
                      )

    ax[0].set_ylabel('Accuracy', labelpad=10.0, fontweight='bold')

    #ax[0].set_xticks([x for x in range(len(labels))])
    #ax[0].set_xticklabels(labels)

    if isinstance(accs, dict):
        number_lines = len(list(accs.keys()))
        sorted_accs_keys = sorted(list(accs.keys()))
    else:
        number_lines = accs.shape[0]
        sorted_accs_keys = list(range(number_lines))

    if len(sorted_accs_keys) < 10:
        ax[0].set_xticks([i+0.5 for i in range(len(sorted_accs_keys))])
        ax[1].set_xticks([i+0.5 for i in range(len(sorted_accs_keys))])
        names = sorted_accs_keys if isinstance(accs, dict) else [labels[i] for i in sorted_accs_keys]
        names = [n.replace(' ', '\n').replace('_', '\n').\
                 replace('w2v', 'Word2Vec').replace('tr', 'Tr').\
                 replace('wikipedia2vec', 'Wikipedia\n2Vec').\
                 replace('\noriginal', '') for n in names]
        ax[0].set_xticklabels(names)
    else:
        ax[0].tick_params(axis='x', which='both', bottom=False, top=False, \
                          labelbottom=False, labeltop=False)
        positions_one = [0.5, 2.5, 5.5, 8.5]
        level_one = ['Overall', 'Categories', 'Individual entities', 'Category vs entity']
        for p, t in zip(positions_one, level_one):
            ax[0].text(s=t, x=p, y=1.2, fontweight='bold', ha='center')
        for n_i, n in enumerate(sorted_accs_keys):
            ax[0].text(s='\n'.join(n.replace('_', ' ').\
                                     replace('luogo', 'place').\
                                     replace('persona', 'person').\
                                     replace('Accuracy', '').\
                                     split()[-2:]), x=n_i+0.5, y=1.1, \
                                     fontweight='demibold', \
                                     ha='center', va='center')
        ax[0].hlines(y=1.175, xmin=0.05, \
                     xmax=1., color='gray', \
                     linestyle='dashdot', alpha=0.7, \
                     )
        ax[0].hlines(y=1.175, xmin=1.15, \
                     xmax=3.85, color='gray', \
                     linestyle='dashdot', alpha=0.7, \
                     )
        ax[0].hlines(y=1.175, xmin=4.15, \
                     xmax=6.85, color='gray', \
                     linestyle='dashdot', alpha=0.7, \
                     )
        ax[0].hlines(y=1.175, xmin=7.15, \
                     xmax=9.85, color='gray', \
                     linestyle='dashdot', alpha=0.7, \
                     )

    ### Legend properties
    ax[1].hlines(y=.35, xmin=0, \
    #ax[1].hlines(y=.5, xmin=0, \
                 xmax=len(labels), color='gray', \
                 linestyle='dashdot', alpha=0., \
                 )
    #ax[1].hlines(y=.25, xmin=0, \
    #            xmax=len(labels), color='gray', \
    #            linestyle='dashdot', alpha=0.2, \
    #            )

    ### Removing all the parts surrounding the plot below
    #ax[1].set_xlim(left=0., right=1.)
    #ax[1].set_ylim(bottom=int(number_lines/2)+1.5, top=0.)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    #ax[1].get_yaxis().set_visible(False)
    #ax[1].set_ylabel('Median')
    ax[1].set_yticks([0.35])
    #ax[1].set_yticks([0.05, 0.35])
    ax[1].set_yticklabels(['Average'])
    #ax[1].set_yticklabels(['Median', 'Average'])

    labels = ax[1].get_yticklabels() + ax[0].get_xticklabels()
    [label.set_fontweight('bold') for label in labels]

    if len(sorted_accs_keys) < 10:
        
        ax[1].scatter(x=[1.4,1.5,1.6], y=[.05, .05, .05], marker='*', color='black')
        ax[1].text(s='p<=0.0005', x=2.2, y=.05, ha='center', va='center')
        ax[1].scatter(x=[3.45,3.55], y=[.05, .05], marker='*', color='black')
        #ax0].text(s='p<=0.001', x=3.65, y=-.24)
        ax[1].text(s='p<=0.005', x=4.05, y=.05, ha='center', va='center')
        ax[1].scatter(x=[5.5], y=[.05], marker='*', color='black')
        #ax[0].text(s='p<=0.01', x=5.6, y=-.24)
        ax[1].text(s='p<=0.05', x=6., y=.05, ha='center', va='center')

    else:
        ax[1].scatter(x=[1.4,1.5,1.6], y=[.05, .05, .05], marker='*', color='black')
        ax[1].text(s='p<=0.0005', x=2.4, y=.05, ha='center', va='center')
        ax[1].scatter(x=[4.45,4.55], y=[.05, .05], marker='*', color='black')
        #ax[0].text(s='p<=0.001', x=3.65, y=-.24)
        ax[1].text(s='p<=0.005', x=5.25, y=.05, ha='center', va='center')
        ax[1].scatter(x=[7.5], y=[.05], marker='*', color='black')
        #ax[0].text(s='p<=0.01', x=5.6, y=-.24)
        ax[1].text(s='p<=0.05', x=8.1, y=.05, ha='center', va='center')

    line_counter = 0

    ### FDR correction
    if isinstance(accs, dict):
        p_values = [stats.wilcoxon(accs[key], [.5 for y in accs[key]], \
                   alternative='greater')[1] for key in sorted_accs_keys]
    else:
        p_values = [stats.wilcoxon(data, [.5 for y in data], \
                   alternative='greater')[1] for data in accs]
    _, fdr_corrected = mne.stats.fdr_correction(p_values)
    
    ### Plotting each violin
    for feature_index, name in enumerate(sorted_accs_keys):
        data = numpy.array(accs[name], dtype=numpy.double)
        conf_interval = stats.t.interval(0.95, len(data)-1, loc=numpy.mean(data), scale=stats.sem(data))

        violin = ax[0].violinplot(data, positions=[feature_index+.5], \
                                          showextrema=False, \
                                          showmedians=False)
        for violin_part in violin['bodies']:
            violin_part.set_edgecolor('black')

        ax[0].vlines(feature_index+.5, ymin=min(data), ymax=max(data), colors='black', lw=1)
        first_quartile, third_quartile = numpy.percentile(data, [25, 75])

        ### Quartiles
        #ax[0].vlines(feature_index+.5, ymin=first_quartile, ymax=third_quartile, colors='dimgray', lw=4)
        ### Confidence intervals
        ax[0].vlines(feature_index+.5, ymin=conf_interval[0], ymax=conf_interval[1], colors='dimgray', lw=4)
        ax[0].vlines(feature_index+.5, ymin=first_quartile, ymax=third_quartile, colors='dimgray', lw=4)
        ax[0].scatter(x=feature_index+.5, y=numpy.average(data), s=15, color= 'white', marker='H', \
                     zorder=3)

        ### Plotting the legend in 
        ### a separate figure below
        line_counter += 1
        if line_counter <= int(number_lines / 2):
            x_text = .05
            y_text = line_counter + 0.5
        else:
            x_text = .55
            y_text = line_counter - int(number_lines/2) + 0.5
        #elif line_counter > number_lines*0.66:
        #    x_text = .7
        #    y_text = 0.+line_counter*.033

        #name = name if isinstance(accs, dict) else labels[name]
        p_value = round(fdr_corrected[feature_index], 4)
 
        if p_value <= .0005:
            asterisk_pos = [.4,.5,.6]
        elif p_value <= 0.005:
            asterisk_pos = [.45,.55]
        elif p_value <= 0.05:
            asterisk_pos = [.5]

        if p_value <= 0.05:
            for asterisk in asterisk_pos:
                #ax[0].scatter(x=feature_index+asterisk, y=.95, marker='*', color='black')
                ax[1].scatter(x=feature_index+asterisk, y=.5, marker='*', color='black')

        #ax[1].text(s=round(numpy.median(data), 2), x=feature_index+.5, y=.05, ha='center', va='center')
        ax[1].text(s=round(numpy.average(data), 3), x=feature_index+.5, y=.35, ha='center', va='center')

        '''
        if p_value == 0.0:
            p_value = 'p<0.0001'
        else:
            p_value = 'p={}'.format(p_value)
        label =  '{} - mean: {} - median {} - {}'.format(name, \
                              #labels[feature_index].replace('stability', ''), \
                              round(numpy.average(data), 2), \
                              round(numpy.median(data),2), \
                              p_value)
        ax[1].scatter(x_text, y_text, \
                      #color=colors[f_i], \
                      label=label, alpha=1.)
        ax[1].text(x_text+.05, y_text, label)
        '''


    pyplot.savefig(os.path.join(plot_path), dpi=600)
    #pyplot.savefig(os.path.join(plot_path), dpi=600)
    pyplot.clf()
    pyplot.close(fig)
