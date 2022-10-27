import mne
import os
import pandas
import tqdm
import matplotlib
import autoreject
#matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot
from multiprocessing import Pool

from tqdm import tqdm

from mne.preprocessing import create_ecg_epochs, create_eog_epochs, ICA, read_ica

from autoreject.utils import interpolate_bads


#for s in range(1, 20):
def preprocess_eeg(s):

    sub_list = list()
    problems = list()

    eeg_folder = 'sub-{:02}/sub-{:02}_eeg'.format(s, s)
    print(eeg_folder)
    events_folder = 'sub-{:02}/sub-{:02}_events'.format(s, s)
    for r in range(1, 25):
        bdf_filename = 'sub-{:02}_run-{:02}.bdf'.format(s, r)
        full_path = os.path.join(main_folder, eeg_folder, bdf_filename)
        full_raw = mne.io.read_raw_bdf(full_path, \
                                      preload=True, \
                                      eog=eog_channels, \
                                      exclude=excluded_channels, \
                                      verbose=False, \
                                      )
        full_raw.set_montage(montage)
        ### Low-pass filtering all data at 0.01-80.hz
        full_raw.filter(l_freq=None, h_freq=80.)

        run_events_full = mne.find_events(full_raw)
        ### Fixing wrong events
        mask = run_events_full[:, 2] < 111
        run_events = run_events_full[mask]

        assert run_events.shape[0] <= 40
        if run_events.shape[0] != 40:
            problems.append('subject {}, run {}, number of trials: {}'.format(s, r, run_events.shape[0]))

        if run_events.shape[0] > 1:
            
            ### Cropping data so as to remove useless data before/after events
            sampling = full_raw.info['sfreq']
            start_point = max(0, int(run_events[0][0]-int(sampling/2)))
            end_point = int(run_events[-1][0]+int(sampling+(sampling/2)))
            run_raw = full_raw.crop(tmin=full_raw.times[start_point], tmax=min(full_raw.times[end_point], full_raw.times[-1]))

            ### Trying again with the events, making sure all is good
            run_events_full = mne.find_events(run_raw)
            ### Fixing wrong events
            mask = run_events_full[:, 2] < 111
            run_events = run_events_full[mask]

            assert run_events.shape[0] <= 40
            if run_events.shape[0] != 40:
                problems.append('subject {}, run {}, number of trials: {}'.format(s, r, run_events.shape[0]))

            ### Filtering eog at 1.-50. hz so as to avoid problems with
            ### autoreject (following the reproducible pipeline on Wakeman & Henson
            picks_eog = mne.pick_types(run_raw.info, eeg=False, eog=True)
            run_raw.filter(l_freq=1., h_freq=50., picks=picks_eog)

            ### Finding the events and reducting the data to epochs
            # Baseline correction is applied here
            run_epochs = mne.Epochs(run_raw, \
                                    run_events, \
                                    tmin=-0.1, \
                                    tmax=1.4, \
                                    #baseline=(None, 0), \
                                    preload=True)

            ### Reducing to a sample rate of 256
            run_epochs.decimate(8)

            ### Autoreject EEG channels, when having more than 10 epochs as cross-validation \
            ### for autoreject is done in 10 folds
            if len(run_epochs) > 10:
                picks_eeg = mne.pick_types(run_epochs.info, eeg=True, eog=False)
                ar = autoreject.AutoReject(n_jobs=os.cpu_count(), \
                                           random_state=1, \
                                           verbose='tqdm', \
                                           picks=picks_eeg)
                run_epochs, autoreject_log = ar.fit_transform(run_epochs, \
                                                              return_log=True)

                reject = autoreject.get_rejection_threshold(run_epochs.copy(), \
                                                            ch_types='eeg')
                run_epochs.drop_bad(reject=reject)

            ### Setting the reference to the average of the channels
            run_epochs.set_eeg_reference(ref_channels='average', \
                                         ch_type='eeg')

            ### Computing and applying ICA to correct EOG
            ### ICA is computed on non-epoched filtered data, 
            ### and applied on unfiltered data
            ica_raw = run_raw.copy().filter(l_freq=1., h_freq=None)
            ica = ICA(n_components=15, \
                      random_state=1)
            ica.fit(ica_raw)
            ### We create the EOG epochs, and then look for the correlation
            ### between them and the individual components
            eog_epochs = create_eog_epochs(run_raw, \
                                           tmin=-.5, \
                                           tmax=.5)
            eog_inds, scores_eog = ica.find_bads_eog(eog_epochs)
            ica.exclude = []
            ica.exclude.extend(eog_inds)
            ica.apply(run_epochs)
            ### Dropping the EOG
            run_epochs.drop_channels(eog_channels)

            ### Applying the baseline - not needed as we're filtering
            #run_epochs.apply_baseline(baseline=(-0.1, 0))

            sub_list.append(run_epochs)

    sub_epochs = mne.concatenate_epochs(sub_list)
    sub_epochs.save(os.path.join(main_folder, eeg_folder, 'sub-{:02}_eeg-epo.fif'.format(s)), \
                   overwrite=True)
    
    return problems

### Channel naming

#eeg_channels = ['{}{}'.format(letter, number) for number in range(1, 33) for letter in ['A', 'B', 'C', 'D']]
eog_channels = ['EXG1', 'EXG2']
excluded_channels = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']
#main_folder = '../../../../OneDrive - Queen Mary, University of London/ProperNamesEEG'
#main_folder = '/import/cogsci/andrea/dataset/neuroscience/ProperNamesEEG'
main_folder = '/import/cogsci/andrea/dataset/neuroscience/FamNamesEEG'

montage = mne.channels.make_standard_montage(kind='biosemi128')

with Pool(processes=os.cpu_count()-1) as pool:
    problems = pool.map(preprocess_eeg, [i for i in range(1, 34)])
    #problems = pool.map(preprocess_eeg, [27])
    pool.close()
    pool.join()

print('Could not get correctly events for:')
print(problems)
