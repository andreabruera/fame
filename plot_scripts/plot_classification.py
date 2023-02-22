import copy
import matplotlib
import mne
import numpy
import os
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats

from io_utils import prepare_folder
from plot_scripts.plot_violins import plot_violins

def return_baseline(args):

    if args.analysis in ['time_resolved_rsa', 'time_resolved_rsa_encoding']:
        random_baseline=0.
    else:
        if args.experiment_id == 'one':
            if args.analysis == 'classification_coarse':
                random_baseline = 0.5
            elif args.analysis == 'classification_fine':
                if args.semantic_category != 'all':
                    random_baseline = 0.25
                else:
                    random_baseline = 0.125
        elif args.experiment_id == 'two':
            random_baseline = 0.5

    return random_baseline

def check_statistical_significance(args, setup_data, times):

    if not type(setup_data) == numpy.array:
        setup_data = numpy.array(setup_data)
    ### Checking for statistical significance
    random_baseline = return_baseline(args)
    ### T-test
    '''
    original_p_values = stats.ttest_1samp(setup_data, \
                         popmean=random_baseline, \
                         alternative='greater').pvalue
    '''
    if args.data_kind not in ['erp', 'alpha', 'beta', 'gamma', 'delta', 'theta']:
        ### Wilcoxon + FDR correction
        significance_data = setup_data.T - random_baseline
        original_p_values = list()
        for t in significance_data:
            p = stats.wilcoxon(t, alternative='greater')[1]
            original_p_values.append(p)

        assert len(original_p_values) == setup_data.shape[-1]
        #corrected_p_values = mne.stats.fdr_correction(original_p_values[2:6])[1]
        #corrected_p_values = original_p_values[:2] +corrected_p_values.tolist() + original_p_values[6:]
        corrected_p_values = mne.stats.fdr_correction(original_p_values)[1]
    else:
        ### TFCE correction using 1 time-point window
        ### following Leonardelli & Fairhall 2019, checking only in the range 100-750ms
        lower_indices = [t_i for t_i, t in enumerate(times) if t<0.1]
        upper_indices = [t_i for t_i, t in enumerate(times) if t>.75]

        relevant_indices = [t_i for t_i, t in enumerate(times) if (t>=0.1 and t<=.75)]
        setup_data = setup_data[:, relevant_indices]
        adj = numpy.zeros((setup_data.shape[-1], setup_data.shape[-1]))
        for i in range(setup_data.shape[-1]):
            #if args.subsample == 'subsample_2' or args.data_kind != 'erp':
            win = range(1, 2)
            #win = range(1, 3)
            #if args.subsample == 'subsample_2':
            #    win = range(1, 2)
            #else:
            #    win = range(1, 3)
            for window in win:
                adj[i, max(0, i-window)] = 1
                adj[i, min(setup_data.shape[-1]-1, i+window)] = 1
        adj = scipy.sparse.coo_matrix(adj)
        corrected_p_values = mne.stats.permutation_cluster_1samp_test(setup_data-random_baseline, tail=1, \
                                                     #n_permutations=4000,
                                                     #adjacency=None, \
                                                     adjacency=adj, \
                                                     threshold=dict(start=0, step=0.2))[2]

        corrected_p_values = [1. for t in lower_indices] + corrected_p_values.tolist() + [1. for t in upper_indices]
    assert len(corrected_p_values) == len(times)
    print(min(corrected_p_values))
    significance = 0.05
    significant_indices = [(i, v) for i, v in enumerate(corrected_p_values) if round(v, 2)<=significance]
    semi_significant_indices = [(i, v) for i, v in enumerate(corrected_p_values) if (round(v, 2)<=0.08 and v>0.05)]
    print('Significant indices at {}: {}'.format(significance, significant_indices))

    return significant_indices, semi_significant_indices

def possibilities(args):

    if args.experiment_id == 'one':

        if args.semantic_category in ['people', 'places']:
            categories = ['people', 'places']
            #categories = [args.semantic_category]
            #entities = ['individuals_only', 'individuals_and_categories']
            entities = [args.entities]

        else:
            categories = [args.semantic_category]
       
            if args.entities == 'individuals_to_categories' or \
                            args.entities == 'all_to_categories':
                entities = ['individuals_to_categories', \
                            #'individuals_only'\
                            ]
                            #entities = [args.entities]
            else:
                entities = [#'individuals_only', 
                            'controlled for length', 'uncontrolled',
                            #'individuals_to_categories', \
                            #'individuals_and_categories', \
                            #'all_to_individuals'
                           ]
                entities = [args.entities]
    else:
        #entities = ['famous_and_familiar']
        categories = ['famous_and_familiar']
        #categories = ['classification_coarse', 'classification_famous_familiar']
        entities = ['classification_coarse', 'classification_famous_familiar']

    if args.corrected:
        if args.semantic_category == 'all':
            controls = [' - {} - controlled for length'.format(args.entities.replace('_', ' '))]
        else:
            controls = [' - controlled for length']
    else:
        if args.semantic_category == 'all':
            controls = [' - {} - uncontrolled'.format(args.entities.replace('_', ' '))]
        else:
            controls = [' - uncontrolled']

    ### ERP
    if args.data_kind != 'time_frequency':

        data_dict = {'' : {c : {k : list() for k in controls} for c in categories}}

    ### Time-frequency
    elif args.data_kind == 'time_frequency':

        frequencies = numpy.arange(1, 40, 3)
        frequencies = numpy.arange(44, 80, 3)
        data_dict = {'{}_hz_'.format(hz) : {c : {k : list() for k in entities} \
                                   for c in categories} for hz in frequencies}

    return data_dict

def read_files(args):

    subjects = 33

    '''
    data_dict = possibilities(args)

    for hz, v in data_dict.items():

        for cat, values in v.items():

            for control in values.keys():
    '''

    data = list()
    path = prepare_folder(args)
    if args.experiment_id == 'one':
        pass
        #path = path.replace(args.semantic_category, cat)
        #    #path = os.path.join('results', args.experiment_id, args.data_kind, args.analysis, \
        #    #                         e_type, args.subsample, 'average_0', args.semantic_category) 
        #path = prepare_folder(args).replace(args.entities, e_type)
        #args_copy = copy.deepcopy(args)
        #args_copy.entities = e_type
        #args_copy.semantic_category = cat
        #path = prepare_folder(args_copy)
        ### Correcting for people/places if needed
        # #path = os.path.join('results', args.experiment_id, args.data_kind, cat, \
        # #                         e_type, args.subsample, 'average_0') 
        #path = prepare_folder(args).replace(args.analysis, e_type).replace(args.entities, cat)

                

    #if 'searchlight' in args.analysis:
    #    file_lambda = lambda arg_list : os.path.join(arg_list[0], \
    #                  '{}sub_{:02}_cluster_{}_accuracy_scores.txt'.format(\
    #                  arg_list[1], arg_list[2], arg_list[3]))
    #else:
    #file_lambda = lambda arg_list : os.path.join(arg_list[0], \
    #              'sub_{:02}_accuracy_scores.txt'.format(\
    #              arg_list[1]))

    #for f in os.listdir(path):
    for sub in range(1, subjects+1):

        #if 'searchlight' in args.analysis:
        #    file_path = file_lambda([path, hz, sub, electrode])
        #else:
        #    file_path = file_lambda([path, hz, sub])
        #if args.corrected:
        #    file_path = file_path.replace('_scores.txt', '_corrected_scores.txt')
        #else:
        #    file_path = file_path.replace('_scores.txt', '_uncorrected_scores.txt')
        correction = 'corrected' if args.corrected else 'uncorrected'
        if args.analysis in ['time_resolved_rsa', 'time_resolved_rsa_encoding']:
            file_path = os.path.join(path, 'sub_{:02}_{}_{}_scores.txt'.format(sub, args.word_vectors, correction))
        else:
            file_path = os.path.join(path, 'sub_{:02}_accuracy_{}_scores.txt'.format(sub, correction))
        
        if not os.path.exists(file_path):
            print('missing: {}'.format(file_path))
            continue
        with open(file_path) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        lines = [l for l in lines if l!=['']]
        ### One line for the times, one for the scores
        assert len(lines) == 2

        #if 'whole_trial' in args.analysis:
        #    lines = numpy.array([v for v in lines[1]], \
        #                             dtype=numpy.double)
        #    times = [0.]
        #    assert lines.shape == (1, )
        #else:

        ### Plotting times until t_max
        t_min = -.05
        if args.experiment_id == 'one':
            t_max = 1.2
        else:
            t_max = 0.9
        times = [float(v) for v in lines[0]]
        times = [t for t in times if t <= t_max]

        ### Collecting subject scores
        lines = [float(v) for v in lines[1]]
        lines = [lines[t_i] for t_i, t in enumerate(times) if t>=t_min]
        times = [t for t in times if t >= t_min]

        '''
        final_lines = list()
        final_times = list()
        for i in range(len(times)):
            if times[i] >= -0.05:
                final_lines.append(lines[i])
                final_times.append(times[i])
        '''
        #data_dict[hz][cat][control].append(lines)
        data.append(lines)
        '''
    times = final_times
    '''
    data = numpy.array(data, dtype=numpy.float32)

    return data, times

def plot_classification(args):

    # Setting font properties

    # Using Helvetica as a font
    font_folder = '/import/cogsci/andrea/dataset/fonts/'
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

    wong_palette = ['goldenrod', 'skyblue', \
                    'mediumseagreen', 'chocolate', \
                    'palevioletred', 'khaki']

    SMALL_SIZE = 23
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 27

    pyplot.rc('font', size=SMALL_SIZE)          # controls default text sizes
    pyplot.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    pyplot.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    pyplot.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    pyplot.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    pyplot.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ### Reading the files
    ### plotting one line at a time, nice and easy

    data, times = read_files(args)

    ### Setting the path

    plot_path = os.path.join('plots', 
                             args.experiment_id, 
                             args.data_kind, 
                             args.analysis,
                             args.entities,
                             args.semantic_category,
                             )
    os.makedirs(plot_path, exist_ok=True)


    ### Preparing a double plot
    fig, ax = pyplot.subplots(nrows=2, ncols=1, \
                              gridspec_kw={'height_ratios': [4, 1]}, \
                              figsize=(16,9), constrained_layout=True)
    #fig.tight_layout()

    ### Main plot properties

    ##### Title (usually not set)
    title = 'Classification scores for {} data\n'\
                    'type of analysis {}'.format(
                    args.data_kind, args.analysis)
    title = title.replace('_', ' ')
    #ax[0].set_title(title)

    ##### Axes
    ax[0].set_xlabel('Time', labelpad=10.0, fontweight='bold')
    if args.analysis == 'time_resolved_rsa':
        ylabel = 'Pearson correlation'
    else:
        ylabel = 'Classification accuracy'
    ax[0].set_ylabel(ylabel, labelpad=10.0, fontweight='bold')

    #### Random baseline line
    random_baseline = return_baseline(args)
    ax[0].hlines(y=random_baseline, xmin=times[0], \
                 xmax=times[-1], color='darkgrey', \
                 linestyle='dashed')

    #### Setting limits on the y axes depending on the number of classes
    if args.semantic_category != 'all' and args.experiment_id=='one':
        ymin = 0.15
        ymax = 0.35
        correction = 0.01
        ax[0].set_ylim(bottom=ymin, top=ymax)
    else:
        if args.analysis in ['time_resolved_rsa', 'time_resolved_rsa_encoding']:
            correction = 0.01
            ymin = -.1
            ymax = .15
        elif args.analysis =='classification_coarse' or args.experiment_id == 'two':
            correction = 0.02
            if args.experiment_id == 'one':
                if args.entities == 'individuals_to_categories':
                    ymin = 0.37
                    ymax = 0.63
                else:
                    ymin = 0.4
                    ymax = 0.6
            elif args.experiment_id == 'two':
                ymin = 0.4
                ymax = 0.63
            ax[0].set_ylim(bottom=ymin, top=ymax)
        elif 'fine' in args.analysis:
            correction = 0.01
            ymin = 0.075
            ymax = 0.175
            ax[0].set_ylim(bottom=ymin, top=ymax)

    ### Plotting when stimulus appears and disappears
    ### using both a line and text
    ax[0].vlines(x=0., ymin=ymin+correction, \
                 ymax=ymax-correction, color='darkgrey', \
                 linestyle='dashed')
    ax[0].text(x=0.012, y=ymin+correction, s='stimulus\nappears', \
                ha='left', va='bottom', fontsize=23)
    #stim_disappears = 0.75 if args.experiment_id=='one' else 0.5
    #ax[0].vlines(x=stim_disappears, ymin=ymin+correction, \
    #             ymax=ymax-correction, color='darkgrey', \
    #             linestyle='dashed')
    #ax[0].text(x=stim_disappears+0.02, y=ymin+correction, s='stimulus\ndisappears', \
    #            ha='left', va='bottom', fontsize=23)
    ### Removing all the parts surrounding the plot above

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ### Setting up the plot below

    ### Fake line just to equate plot 0 and 1
    ax[1].hlines(y=0.15, xmin=times[0], \
                 xmax=times[-1], color='darkgrey', \
                 linestyle='dashed', alpha=0.)
    ### Real line
    ax[1].hlines(y=0.15, xmin=times[0], \
                 xmax=times[-1], color='darkgrey', \
                 linestyle='dashed', alpha=1, linewidth=1.)

    ### Setting limits
    ax[1].set_ylim(bottom=.5, top=.0)

    ### Removing borders
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    #ax[1].spines['left'].set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    #ax[1].get_yaxis().set_visible(False)

    ### Setting p<=0.05 label in bold 
    ax[1].set_yticks([0.15])
    ax[1].set_yticklabels(['p<=0.05'])
    labels = ax[1].get_yticklabels()
    [label.set_fontweight('bold') for label in labels]

    ### Setting colors for plotting different setups
    #cmap = cm.get_cmap('plasma')
    #colors = [cmap(i) for i in range(32, 220, int(220/len(folders)))]

    #number_lines = len([1 for k, v in data_dict.items() \
    #                   for d in v.keys()])

    #line_counter = 0
    #sig_container = dict()

    #import pdb; pdb.set_trace()
    #for cat, values in v.items():

    #    for e_type, subs in values.items():

    #        #if args.semantic_category in ['people', 'places']:
    #        #    color = colors[cat]
    #        #else:
    #        #if args.semantic_category == 'people':
    #        if cat == 'people':
    #            color = 'steelblue'
    #        #elif args.semantic_category == 'places':
    #        elif cat == 'places':
    #            color = 'darkorange'
    #        #elif args.semantic_category == 'familiar':
    #        elif cat == 'familiar':
    #            color = 'mediumspringgreen'
    #        #elif args.semantic_category == 'famous':
    #        elif cat == 'famous':
    #            color = 'magenta'
    #        else:
    #            #color = colors[e_type]
    #            #color = 'goldenrod'

    ### Plot is randomly colored
    color = (numpy.random.random(), numpy.random.random(), numpy.random.random())

    sig_values, semi_sig_values = check_statistical_significance(args, data, times)
    sig_indices = [k[0] for k in sig_values]
    semi_sig_indices = [k[0] for k in semi_sig_values]

    assert len(data) == 33
    '''
    if args.experiment_id == 'one':
        label = '{}{}'.format(cat, e_type).replace(\
                                           '_', ' ')
    elif args.experiment_id == 'two':
        if args.analysis == 'classification_coarse':
            split_label = 'people vs places'
        elif args.analysis == 'classification_famous_familiar':
            split_label = 'famous vs familiar'
        if args.corrected:
            split_label = '{} - corrected for length'.format(split_label)
        else:
            split_label = '{} - uncorrected'.format(split_label)
        label = '{} - N={}'.format(split_label, len(subs))
        if args.semantic_category in ['people', 'places',
                                'famous', 'familiar']:
            label =  label.replace(split_label, '{}, {}'.format(split_label, args.semantic_category))                    
    '''
    ### Building the label

    ### Starting point
    if 'coarse' in args.analysis:
        label = 'people vs places'
    elif 'fine' in args.analysis:
        label = 'fine-grained classification'
    elif 'famous' in args.analysis:
        label = 'famous vs familiar'
    elif args.analysis == 'time_resolved_rsa':
        label = 'RSA - {}'.format(args.entities)
    elif args.analysis == 'time_resolved_rsa_encoding':
        label = 'RSA - {}'.format(args.entities)
    else:
        raise RuntimeError('There was a mistake')

    ### entities and correction
    
    #if args.analysis in ['time_resolved_rsa' 'time_resolved_rsa_encoding']:
    #    correction = args.word_vectors
    #else:
    correction = 'corrected' if args.corrected else 'uncorrected'
    label = '{} - {} - {}'.format(label, args.semantic_category, correction)

    #if 'whole_trial' in args.analysis:
    #    continue_marker = False
    #    title = 'Accuracy on {}\n{}'.format(args.analysis.\
    #                                  replace('_', ' '), \
    #                                  args.entities)
    #    accs = numpy.array([[v[0] for v in subs]])
    #    labels = [e_type]
    #    
    #    plot_violins(accs, labels, plot_path, \
    #                 title, random_baseline)
    #
    #else:
    #    continue_marker = True

    ### Averaging the data
    average_data = numpy.average(data, axis=0)

    ### Computing the SEM
    sem_data = stats.sem(data, axis=0)

    #if args.subsample == 'subsample_2':
    #    ### Interpolate averages
    #    inter = scipy.interpolate.interp1d(times, \
    #                    average_data, kind='cubic')
    #    x_plot = numpy.linspace(min(times), max(times), \
    #                            num=500, endpoint=True)
    #    #ax[0].plot(times, average_data, linewidth=.5)
    #    ax[0].plot(x_plot, inter(x_plot), linewidth=1., \
    #              color=color)
    #else:

    ### Plotting the average
    ax[0].plot(times, average_data, linewidth=1.,
              color=color)
    #ax[0].errorbar(x=times, y=numpy.average(subs, axis=0), \
                   #yerr=stats.sem(subs, axis=0), \
                   #color=colors[f_i], \
                   #elinewidth=.5, linewidth=1.)
                   #linewidth=.5)

    ### Plotting the SEM
    ax[0].fill_between(times, average_data-sem_data, \
                       average_data+sem_data, \
                       alpha=0.05, color=color)
    #for t_i, t in enumerate(times):
        #ax[0].violinplot(dataset=setup_data[:, t_i], positions=[t_i], showmedians=True)
    
    ### Plotting statistically significant time points
    ax[0].scatter([times[t] for t in sig_indices], \
    #ax[0].scatter(significant_indices, \
               [numpy.average(data, axis=0)[t] \
                    for t in sig_indices], \
                    color='white', \
                    edgecolors='black', \
                    s=20., linewidth=.5)
    ax[0].scatter([times[t] for t in semi_sig_indices], \
    #ax[0].scatter(significant_indices, \
               [numpy.average(data, axis=0)[t] \
                    for t in semi_sig_indices], \
                    color='white', \
                    edgecolors='black', \
                    s=20., linewidth=.5,
                    marker='*')

    ### Plotting the legend in 
    ### a separate figure below
    line_counter = 1
    step = int(len(times)/7)
    if line_counter == 1:
        p_height = .1
        x_text = times[::step][0]
        y_text = 0.4
    elif line_counter == 2:
        p_height = .2
        x_text = times[::step][4]
        y_text = 0.4
    if line_counter == 3:
        x_text = .1
        y_text = 0.1
    if line_counter == 4:
        x_text = .5
        y_text = 0.1
    '''
    if line_counter <= number_lines/3:
        x_text = .1
        y_text = 0.+line_counter*.1
    elif line_counter <= number_lines/3:
        x_text = .4
        y_text = 0.+line_counter*.066
    elif line_counter > number_lines/3:
        x_text = .7
        y_text = 0.+line_counter*.033
    '''

    ax[1].scatter([times[t] for t in sig_indices], \
    #ax[0].scatter(significant_indices, \
               [p_height for t in sig_indices], \
                    s=60., linewidth=.5, color=color)
    ax[1].scatter(x_text, y_text, \
                  #color=colors[f_i], \
                  s=180., color=color, \
                  label=label, alpha=1.,\
                  marker='s')
    ax[1].text(x_text+0.02, y_text, label, \
              fontsize=27., fontweight='bold', \
              ha='left', va='center')

    '''
    for sub in subs:
        ax[0].plot(times, sub, alpha=0.1)
    '''
    #sig_container[label] = sig_values

    #if continue_marker:

    ### Writing to file the significant points
    #if 'searchlight' in args.analysis:
    #    file_name = os.path.join(plot_path,\
    #                   'cluster_{}_{}_{}_{}_{}_{}_{}.txt'.format(electrode, cat, args.analysis, e_type,\
    #                    args.data_kind, hz, args.subsample))
    #else:
    file_name = os.path.join(plot_path,\
                   '{}_{}_{}_{}_{}_average{}_{}.txt'.format(
                            args.analysis, 
                            args.semantic_category,
                            args.entities,
                            args.data_kind, 
                            #args.subsample,
                            args.temporal_resolution,
                            args.average,
                            correction))
    if args.analysis in ['time_resolved_rsa', 'time_resolved_rsa_encoding']:
        file_name = file_name.replace('.txt', '_{}.txt'.format(args.word_vectors))
    with open(file_name, 'w') as o:
        o.write('Data\tsignificant time points & FDR-corrected p-value\n')
        #for l, values in sig_container.items():
        #    o.write('{}\t'.format(l))
        o.write('{}\n\n'.format(label))
        for v in sig_values:
            o.write('{}\t{}\n'.format(times[v[0]], round(v[1], 5)))

    ### Plotting
    #if 'searchlight' in args.analysis:
    #    plot_name = os.path.join(plot_path,\
    #                   #'{}_{}_{}_{}_{}_{}.pdf'.format(cat, args.analysis, e_type, \
    #                   'cluster_{}_{}_{}_{}_{}_{}_{}.jpg'.format(electrode, cat, args.analysis, e_type, \
    #                    args.data_kind, hz, args.subsample))
    #else:
    #    plot_name = os.path.join(plot_path,\
    #                   #'{}_{}_{}_{}_{}_{}.pdf'.format(cat, args.analysis, e_type, \
    #                   '{}_{}_{}_{}_{}_{}.jpg'.format(cat, args.analysis, e_type, \
    #                    args.data_kind, hz, args.subsample))
    plot_name = file_name.replace('txt', 'jpg')
    print(plot_name)
    pyplot.savefig(plot_name, dpi=600)
    pyplot.savefig(plot_name.replace('jpg', 'svg'), dpi=600)
    pyplot.clf()
    pyplot.close(fig)
