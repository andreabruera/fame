import os
import mne
import pandas
import random
import re
import collections
import numpy
import matplotlib
import itertools
import random
import scipy

from tqdm import tqdm
from matplotlib import pyplot
from scipy import stats

from lab.utils import read_words, read_trigger_ids, select_words
from searchlight.searchlight_utils import SearchlightClusters
#from word_vector_enc_decoding.read_word_vectors import compute_clusters

def tfr_frequencies(args):
    ### Setting up each frequency band
    if args.data_kind == 'alpha':
        frequencies = list(range(8, 15))
    elif args.data_kind == 'beta':
        frequencies = list(range(14, 31))
    elif args.data_kind == 'lower_gamma':
        frequencies = list(range(30, 51))
    elif args.data_kind == 'higher_gamma':
        frequencies = list(range(50, 81))
    elif args.data_kind == 'delta':
        frequencies = list(range(1, 5)) + [0.5]
    elif args.data_kind == 'theta':
        frequencies = list(range(4, 9))
    elif args.data_kind == ('erp'):
        frequencies = 'na'
    frequencies = numpy.array(frequencies)

    return frequencies

def prepare_folder(args):

    out_path = os.path.join('results', args.experiment_id, args.data_kind, args.analysis, \
                             args.entities, args.subsample, 'average_{}'.format(args.average), \
                             args.semantic_category, args.wv_dim_reduction)
    os.makedirs(out_path, exist_ok=True)

    return out_path

class ExperimentInfo:

    def __init__(self, args, subject=1):
        
        self.experiment_id = '{}'.format(args.experiment_id)
        self.analysis = args.analysis
        self.entities = args.entities
        self.semantic_category = args.semantic_category
        self.data_folder = args.data_folder
        self.corrected = args.corrected
        self.runs = 24
        self.subjects = 33
        self.current_subject = subject
        self.eeg_paths = self.generate_eeg_paths()
        self.events_log, self.trigger_to_info = self.read_events_log()
        if 'coding' not in self.analysis and 'rsa' not in self.analysis:
            self.test_splits = self.generate_test_splits()

    def generate_eeg_paths(self):
        eeg_paths = dict()
        fold = 'derivatives'
        for s in range(1, self.subjects+1):
            sub_path = os.path.join(
                                    self.data_folder,
                                    fold,
                                    'sub-{:02}'.format(s),
                                    'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(s)
                                    )
            assert os.path.exists(sub_path) == True
            eeg_paths[s] = sub_path
        return eeg_paths

    def read_events_log(self):

        full_log = dict()

        for s in range(1, self.subjects+1):
            events_path = os.path.join(self.data_folder,
                                    'derivatives',
                                    'sub-{:02}'.format(s),
                                    'sub-{:02}_task-namereadingimagery_events.tsv'.format(s))
            assert os.path.exists(events_path) == True
            with open(events_path) as i:
                sub_log = [l.strip().split('\t') for l in i.readlines()]
            sub_log = {h : [l[h_i].strip() for l in sub_log[1:]] for h_i, h in enumerate(sub_log[0])}
            n_trials = list(set([len(v) for v in sub_log.values()]))
            assert len(n_trials) == 1
            ### Initializing the dictionary
            if len(full_log.keys()) == 0:
                full_log = sub_log.copy()
                full_log['subject'] = list()
            ### Adding values
            else:
                for k, v in sub_log.items():
                    full_log[k].extend(v)
            ### adding subject
            for i in range(n_trials[0]):
                full_log['subject'].append(s)
        ### Creating the trigger-to-info dictionary
        trig_to_info = dict()
        for t_i, t in enumerate(full_log['value']):
            ### Limiting the trigger-to-info dictionary to the current subject
            if full_log['subject'][t_i] == self.current_subject:
                name = full_log['trial_type'][t_i]
                if self.experiment_id == 'two':
                    key_one = 'semantic_domain'
                    key_two = 'familiarity'
                else:
                    key_one = 'coarse_category'
                    key_two = 'fine_category'
                cat_one = full_log[key_one][t_i]
                cat_two = full_log[key_two][t_i]
                infos = [name, cat_one, cat_two]
                t = int(t)
                if t in trig_to_info.keys():
                    assert trig_to_info[t] == infos
                trig_to_info[t] = infos

        ### Filtering trig_to_info as required

        ### No correction
        if self.semantic_category == 'all':
            pass
        ### Only a nested sub-class
        else:
            if self.semantic_category in ['people', 'places']:
                cat_index = 1
                filter_word = 'person' if self.semantic_category=='people' else 'place'
            elif self.semantic_category in ['famous', 'familiar']:
                cat_index = 2
                filter_word = '{}'.format(self.semantic_category)
            trig_to_info = {k : v for k, v in trig_to_info.items() if v[cat_index]==filter_word}
        if self.entities in ['people_only', 'places_only']:
            if self.experiment_id == 'one':
                raise RuntimeError('For experiment one there is no people/place only entities choice!')
            else:
                cat_index = 1
                filter_word = 'person' if self.entities=='people_only' else 'place'
                trig_to_info = {k : v for k, v in trig_to_info.items() if v[cat_index]==filter_word}

        ### Using only individuals (experiment one)
        if self.experiment_id == 'one':
            if self.entities == 'individuals_only':
                trig_to_info = {k : v for k, v in trig_to_info.items() if k<=100}

        return full_log, trig_to_info

    def generate_test_splits(self):

        ### Experiment two, somewhat easier
        ### as there are always two classes
        if self.experiment_id == 'two':
            ### Creating for each test set two tuples, one for each class

            ### Reducing stimuli to nested sub-class
            if self.semantic_category in ['people', 'places', 'famous', 'familiar']:
                cat_index = 2
                cat_length = 8
                combinations_one_cat = list(itertools.combinations(list(range(cat_length)), 1))
                combinations = list(itertools.product(combinations_one_cat, repeat=2))
            ### Using all stimuli
            else:
                cat_length = 16
                combinations_one_cat = list(itertools.combinations(list(range(cat_length)), 2))
                combinations = list(itertools.product(combinations_one_cat, repeat=2))
            ### Semantic domain
            if 'coarse' in self.analysis:
                cat_index = 1
            ### Type of familiarity
            else:
                cat_index = 2

        ### Experiment one, there we go...
        else:
            ### Transfer classification (nice 'n easy...)
            if self.entities == 'individuals_to_categories':
                ### Creating for each test set two tuples, one for each class
                if 'coarse' in self.analysis:
                    cat_index = 1
                    cat_length = 4
                    combinations_one_cat = list(itertools.combinations(list(range(cat_length)), 2))
                    combinations = list(itertools.product(combinations_one_cat, repeat=2))
                ### When looking at fine-grained categories, 
                ### there's only one test set with the 4/8 classes
                elif 'fine' in self.analysis:
                    cat_index = 2
                    cat_length = 1
                    n_classes = 8 if self.semantic_category=='all' else 4
                    combinations = [(0 for i in range(n_classes))]
            ### Standard classification
            else:
                ### People vs places
                if 'coarse' in self.analysis:
                    cat_index = 1
                    ### Using individuals only
                    if self.entities == 'individuals_only':
                        cat_length = 16
                    ### Mixing individuals and categories
                    elif self.entities == 'individuals_and_categories':
                        cat_length = 20
                    combinations_one_cat = list(itertools.combinations(list(range(cat_length)), 2))
                    combinations = list(itertools.product(combinations_one_cat, repeat=2))
                ### Fine-grained categories
                elif 'fine' in self.analysis:
                    cat_index = 2
                    if self.entities == 'individuals_only':
                        cat_length = 4
                    elif self.entities == 'individuals_and_categories':
                        cat_length = 5
                    ### Transfer classification
                    else:
                        cat_length = 1
                    ### Using all stimuli
                    if self.semantic_category == 'all':
                        combinations = list(itertools.product(list(range(cat_length)), repeat=8))
                    ### Using only a nested sub-class
                    else:
                        combinations = list(itertools.product(list(range(cat_length)), repeat=4))

        ### Getting the list of classes to be used
        cats = set([v[cat_index] for k, v in self.trigger_to_info.items()])
        ### Transfer classification requires to reduce the possible test triggers
        if self.entities == 'individuals_to_categories':
            cat_to_trigger = {cat : [t for t, info in self.trigger_to_info.items() if info[cat_index] == cat and t>100] for cat in cats}
        ### Standard classification considers them all
        else:
            cat_to_trigger = {cat : [t for t, info in self.trigger_to_info.items() if info[cat_index] == cat] for cat in cats}

        ### Just checking all's good and fine
        for k, v in cat_to_trigger.items():
            assert len(v) == cat_length

        ### Randomizing all test splits
        test_permutations = list(random.sample(combinations, k=len(combinations)))

        ### Creating test splits, and correcting them for length if required
        test_splits = list()
        for i_p, p in enumerate(test_permutations):
            triggers = list()
            ### Fine-grained categories requires a separate case
            if 'fine' in self.analysis:
                if self.semantic_category != 'all':
                    assert len(cat_to_trigger.keys()) == 4
                else:
                    assert len(cat_to_trigger.keys()) == 8
                for t_i, ts in zip(p, cat_to_trigger.values()):
                    triggers.append(ts[t_i])

                ### Checking
                if self.semantic_category != 'all':
                    assert len(triggers) == 4
                else:
                    assert len(triggers) == 8 
            ### Cases where there's only two classes
            else:
                assert len(cat_to_trigger.keys()) == 2
                for kv_i, kv in enumerate(cat_to_trigger.items()):
                    for ind in p[kv_i]:
                        triggers.append(kv[1][ind])

                ### Checking
                if self.semantic_category != 'all':
                    assert len(triggers) == 2
                else:
                    assert len(triggers) == 4 

            test_splits.append(triggers)
        ### Difference across people and places / familiar and unfamiliar
        if 'coarse' in self.analysis:

            stat_diff = [[len(n[0]) for n in self.trigger_to_info.values() if n[1]==k] for k in set([v[1] for v in self.trigger_to_info.values()])]
            stat_diff = scipy.stats.ttest_ind(stat_diff[0], stat_diff[1])
            print('Difference between lengths for people and places: {}'.format(stat_diff))

        ### Collecting average lengths for the current labels

        ### To correct both for coarse and fine-grained lengths
        ### use this line
        ### For replication of experiment one, use this line
        #if 'coarse' in args.analysis or 'fine' in args.analysis:
        if 'coarse' in self.analysis:
            cat_index = 1
        else:
            cat_index = 2
        current_cats = set([v[cat_index] for v in self.trigger_to_info.values()])
        cat_to_average_length = {k : numpy.average([len(n[0]) for n in self.trigger_to_info.values() if n[cat_index]==k]) for k in current_cats}
        ### Replication
        if self.experiment_id == 'one':
            cat_to_average_length = {'actor' : 14,
                                  'musician' : 9,
                                  'writer' : 13,
                                  'politician' : 13,
                                  'person' : 12, 
                                  'place' : 9,
                                  'city' : 6, 
                                  'country' : 7,
                                  "body_of_water" : 12, 
                                  'monument' : 11
                                  }
        print('Current categories and average lengths: {}'.format(cat_to_average_length))

        ### Computing correlation
        split_corrs = list()
        for trigs in test_splits:
            labels = list()
            lengths = list()
            ### Test set
            for t in trigs:
                labels.append(cat_to_average_length[self.trigger_to_info[t][cat_index]])
                lengths.append(len(self.trigger_to_info[t][0]))
            corr = list(scipy.stats.pearsonr(lengths, labels))
            split_corrs.append(corr)

        ### Sorting by increasing absolute value
        split_corrs = sorted(enumerate(split_corrs), key=lambda item : abs(item[1][0]))
        n_folds = 100

        zero_corr_sets = len([t for t in split_corrs if t[1][0]==0.0])
        if self.corrected:
            if zero_corr_sets >= n_folds:
                test_splits = random.sample([test_splits[t[0]] for t in split_corrs if t[1][0]==0.0], k=n_folds)
            else:
                test_splits = [test_splits[t[0]] for t in split_corrs][:n_folds]
        else:
            test_splits = random.sample(test_splits, k=min(n_folds, len(test_splits)))
        print('Minimum correlation within the test set: {}'.format(split_corrs[min(n_folds-1, len(test_splits)-1)]))

        return test_splits

class LoadEEG:
   
    def __init__(self, args, experiment, subject, ceiling=False):

        self.subject = subject
        self.data_path = experiment.eeg_paths[subject]
        self.experiment = experiment
        self.ceiling = ceiling
        self.full_data, self.data_dict, self.times, self.all_times, \
        self.frequencies, self.data_shape = self.load_epochs(args)

    def load_epochs(self, args):

        coll_data_dict = collections.defaultdict(list)
        epochs = mne.read_epochs(
                               self.data_path, 
                               preload=True, 
                               verbose=False
                               )
        ### Restricting data to EEG
        epochs = epochs.pick_types(eeg=True)

        ### Checking baseline correction is fine
        if not epochs.baseline:
            epochs.apply_baseline(baseline=(None, 0))
        else:
            assert len(epochs.baseline) ==2
            assert round(epochs.baseline[0], 1) == -0.1
            assert epochs.baseline[1] == 0.

        ### For decoding, considering only time points after 150 ms
        if args.analysis in ['decoding', 'encoding', 'feature_selection']:
            tmin = 0.15
            tmax = 1.2
            epochs.crop(tmin=tmin, tmax=tmax)
        times = epochs.times
        all_times = times.copy()
        
        ### Transforming to numpy array
        epochs_array = epochs.get_data()
        assert epochs_array.shape[1] == 128

        ### Setting some time-frequency variables
        decimation = 1
        sampling_frequency = epochs.info['sfreq']
        frequencies = tfr_frequencies(args)
        ### Converting to / loading time frequency data
        if args.data_kind != 'erp':
            n_cycles = frequencies / 2

            ### Transforming data into TFR using morlet wavelets
            print('Now transforming into time-frequency...')
            
            epochs_array = mne.time_frequency.tfr_array_morlet(
                                            epochs_array, 
                                            sfreq=sampling_frequency,
                                            decim=decimation,
                                            freqs=frequencies, 
                                            n_cycles=n_cycles, 
                                            n_jobs=int(os.cpu_count()/3),
                                            output='power'
                                            )
            ### Averaging the band
            epochs = numpy.average(epochs_array, axis=2)
            assert epochs.shape[1] == 128

            ### Decimating times
            if decimation > 1:
                times = times[::decimation]
                all_times = times.copy()[::decimation]
            ### Checking
            assert len(times) == epochs_array.shape[-1]

            ### Baseline correct the tfr data (subtracts the mean, takes the log (dB conversion))
            #tfr_epochs = mne.baseline.rescale(epochs_array, times, 
            #                                  baseline=(min(times), 0.),
            #                                  mode='logratio', 
            #                                  )

        ### Scaling 
        epochs_array = mne.decoding.Scaler(epochs.info, \
                    scalings='mean'\
                    ).fit_transform(epochs_array)
        ### Collecting the data shape
        data_shape = epochs_array.shape[1:]

        ### organizing to a dictionary having trigger as key
        ### numpy array or ERPs as values
        full_data_dict = collections.defaultdict(list)
        for i, e in enumerate(epochs.events[:, 2]):
            full_data_dict[int(e)].append(epochs_array[i])
        full_data_dict = {k : numpy.array(v) for k, v in full_data_dict.items()}

        ### Subsampling by average by sub_amount
        if 'subsample' in args.subsample:
            sub_amount = int(args.subsample.split('_')[-1])
            sub_indices = list(range(len(times)))[::sub_amount][:-1]
            for k, epo in full_data_dict.items():
                sub_list = list()
                rolled_average = numpy.array([numpy.average(epo[:, :, i:i+sub_amount], axis=2) for i in sub_indices])
                sub_epo = numpy.moveaxis(rolled_average, 0, 2)
                full_data_dict[k] = sub_epo
            ### Updating times
            times = [times[i+int(sub_amount/2)] for i in sub_indices]

        ### Reducing the number of ERPs by averaging, if required

        # 0: takes the min-max number of ERPs present across all stimuli
        min_value_to_use = min([len(v) for k, v in full_data_dict.items()])
        stimuli_values = {k : len(v) for k, v in full_data_dict.items()}
        print(
              'Min-max amount of available ERPs: {} - median&std of max across stimuli: {}, {}'.format(
                       min_value_to_use, 
                       numpy.median(list(stimuli_values.values())), 
                       numpy.std(list(stimuli_values.values())))
             )
        ### Using min-max of ERPs across stimuli
        if args.average == 0:
            stimuli_values = {k : min_value_to_use for k in stimuli_values.keys()}
        ### Using a fixed amount of ERPs
        else:
            stimuli_values = {k : min(v, args.average) for k, v in stimuli_values.items()}
        data_dict = dict()
        for k, v in full_data_dict.items():
            ### averaging
            averaged_v = numpy.average(
                                       random.sample(v.tolist(), 
                                       k=stimuli_values[k]), 
                                       axis=0
                                       )
            assert averaged_v.shape == v[0].shape
            data_dict[k] = averaged_v

        ### Reducing EEG to trig_to_info keys
        full_data_dict = {k : v for k, v in full_data_dict.items() if k in self.experiment.trigger_to_info.keys()}
        data_dict = {k : v for k, v in data_dict.items() if k in self.experiment.trigger_to_info.keys()}

        return full_data_dict, data_dict, times, all_times, frequencies, data_shape
