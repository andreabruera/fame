import argparse
import datetime
import mne
import os
import random
import re
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--write_rawdata', action='store_true', required=False,
                    help='Writing also EEGLAB\'s raw data?')
args = parser.parse_args()

assert os.path.exists(args.folder)

eog_channels = ['EXG1', 'EXG2']
excluded_channels = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7',
                     'EXG8', 'GSR1', 'GSR2',
                    'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']

### Creating basic output
out_folder = os.path.join('../../exploring_individual_entities_eeg')
os.makedirs(out_folder, exist_ok=True)

### Source data
source_folder = os.path.join(out_folder, 'sourcedata')
os.makedirs(source_folder, exist_ok=True)

### Raw data
raw_folder = os.path.join(out_folder, 'rawdata')
os.makedirs(raw_folder, exist_ok=True)

### Count subjects

subjects = [int(f.replace('sub-', '')) for f in os.listdir(args.folder)]
assert len(subjects) == 33

collector = dict()

category_mapper = {'musicista' : ['person', 'musician'],
                   'attore' : ['person', 'actor'],
                   'scrittore' : ['person', 'writer'],
                   'politico' : ['person', 'politician'],
                   'monumento' : ['place', 'monument'],
                   'città' : ['place', 'city'],
                   'stato' : ['place', 'country'],
                   "corso d'acqua" : ['place', 'body_of_water']
                   }

mapper = {'persona' : 'person',
          'luogo' : 'place',
          'musicista' : 'musician',
          'attore' : 'actor',
          'scrittore' : 'writer',
          'politico' : 'politician',
          'città' : 'city',
          'stato' : 'country',
          'monumento' : 'monument',
          "corso d'acqua" : 'body_of_water'
          }
### Experiment one
with open('exp_one_stimuli.txt') as i:
    lines = [l.strip().split('\t') for l in i.readlines()][1:]
trig_to_cats = {int(l[0]) : [mapper[l[2]], mapper[l[3]]] for l in lines if l[2] in mapper.keys() and l[3] in mapper.keys()}
trig_to_names = {int(l[0]) : l[1] for l in lines}
n_ev = 40
assert len(trig_to_cats.keys()) == n_ev*2

### start with subjects
for s_f in os.listdir(args.folder):
    print(s_f)
    assert len(os.listdir(os.path.join(args.folder, s_f))) == 2
    eeg_folder = os.path.join(args.folder, s_f, '{}_eeg'.format(s_f))
    assert os.path.exists(eeg_folder)
    events_folder = os.path.join(args.folder, s_f, '{}_events'.format(s_f))
    assert os.path.exists(events_folder)


    ### eeg data
    ### check files are all there
    e_fs = [f for f in os.listdir(eeg_folder) if 'bdf' in f]
    assert len(e_fs) == 24
    e_f_ids = [int(re.findall('(\d\d)(?=[.])', f)[0]) for f in e_fs]
    assert min(e_f_ids) == 1
    assert max(e_f_ids) == 24

    s_source = os.path.join(source_folder, s_f)
    os.makedirs(s_source, exist_ok=True)
    s_raw = os.path.join(raw_folder, s_f)
    os.makedirs(s_raw, exist_ok=True)
    for f in e_fs:
        original_path = os.path.join(eeg_folder, f)
        #print(original_path)
        assert os.path.exists(original_path)
        ### Copying to source data
        out_file = f.replace('run', 'task-namereadingimagery_run')
        out_file = out_file.replace('.', '_eeg.')
        #print('cp {} {}'.format(original_path, os.path.join(s_source, out_file)))
        #os.system('cp '+os.path.join(args.folder, s_f, f)+' '+s_source+'/')
        #os.system('mv {} {}'.format(os.path.join(s_source, f), os.path.join(s_source, out_file)))
        shutil.copyfile(original_path, os.path.join(s_source, out_file))
        ### Converting to EDF raw data
        raw_f = mne.io.read_raw(original_path,
                                    eog=eog_channels,
                                    exclude=excluded_channels,
                                    #misc=['Status'],
                                    verbose=False,
                                    preload=True)
        raw_f.info['line_freq'] = 50
        sub_n = int(re.findall('(\d\d)(?=_eeg)', out_file)[0])
        raw_f.info['subject_info'] = {'id' : sub_n,
                                      'sex' : 0}

        ### Setting montage
        montage = mne.channels.make_standard_montage('biosemi128')
        raw_f.set_montage(montage)

        ### Removing events which were not meant to be recorded
        events = mne.find_events(raw_f,
                                 initial_event=False,
                                 verbose=False,
                                 stim_channel='Status',
                                 min_duration=0.5
                                 )
        max_trigger = 111

        if events.shape[0] < n_ev:
            print(events.shape[0])
            print('{} missing some'.format(out_file))
        elif events.shape[0] > n_ev:
            print('{} having too many'.format(out_file))
            ### Correcting

        to_be_removed = events[[i_i for i_i, i in enumerate(events[:, 2] < max_trigger) if i == True], :]
        raw_f.add_events(to_be_removed, replace=True, stim_channel='Status')
        events = mne.find_events(raw_f,
                                 initial_event=False,
                                 stim_channel='Status',
                                 verbose=False)
        if events.shape[0] > n_ev:
            print('{} having too many'.format(out_file))
        if events.shape[0] < n_ev:
            print(mne.find_events(raw_f, verbose=False).shape[0])
            print('{} missing some'.format(out_file))

        ### Cropping so as to make it smaller
        if events.shape[0] > 0:
            ### keeping 0.5s before
            #min_t = raw_f.times[max(0, events[0][0]-1024)]
            ### keeping 2s after last stimulus
            max_t = raw_f.times[min(len(raw_f.times)-1, events[-1][0]+4096)]
            raw_f.crop(#tmin=min_t,
                       tmax=max_t
                       )

        raw_out_file = os.path.join(s_raw, out_file.replace('.bdf', '.set'))
        source_out_file = os.path.join(s_source, out_file.replace('.bdf', '.set'))
        for e in events[:, 2]:
            assert e < max_trigger
        collector[source_out_file] = [{k : t for k, t in zip(events[:,2],
            [raw_f.times[e] for e in events[:, 0]])}, events[:, 2]]
        if args.write_rawdata:
            mne.export.export_raw(raw_out_file, raw_f, fmt='eeglab',
                                  overwrite=True, add_ch_type=True)

    ### events data
    ### check files are all there
    e_fs = [f for f in os.listdir(events_folder) if 'pkl' not in f]
    assert len(e_fs) == 24
    e_f_ids = [int(re.findall('(\d\d)(?=[.])', f)[0]) for f in e_fs]
    assert min(e_f_ids) == 0
    assert max(e_f_ids) == 23
    for f in e_fs:
        original_f = os.path.join(events_folder, f)
        assert os.path.exists(original_f)
        out_file = f.replace('run', 'task-namereadingimagery_run').replace('.events', '.tsv')
        ### Correcting number
        run_n = int(re.findall('(\d\d)(?=[.])', f)[0])
        out_file = out_file.replace('run_{:02}'.format(run_n), 'run-{:02}'.format(run_n+1))
        out_file = '{}_{}'.format(s_f, out_file)
        out_file = out_file.replace('.tsv', '_events.tsv')
        #print('cp {} {}'.format(original_f, os.path.join(s_source, out_file.format('events', 'events_original'))))
        shutil.copyfile(original_f, os.path.join(s_source, out_file.replace('events', 'events_original')))
        #os.system('cp {} {}'.format(os.path.join(args.folder, s_f, f), os.path.join(s_source, out_file)))

        ### Writing the BIDS events file
        with open(original_f) as i:
            lines = [l.split('\t') for l in i.readlines()][1:]
        for l in lines:
            if int(l[1]) not in trig_to_names.keys():
                trig_to_names[int(l[1])] = l[0]
            if int(l[1]) not in trig_to_cats.keys():
                trig_to_cats[int(l[1])] = category_mapper[trig_to_names[int(l[1])]]
        ### 0 : word, 1 : trigger,
        ### 2 : question type, 3 : correct answer, 4 : answer given,
        ### 5 : reaction time
        events_out = os.path.join(s_source, out_file)

        ### First, we need to align recordings and events
        recorded_events = collector[events_out.replace('events.tsv', 'eeg.set')][0]

        with open(events_out, 'w') as o:
            ### proposed by BIDS
            o.write('onset\tduration\ttrial_type\tvalue\tresponse_time\t')
            ### Dataset-specific
            o.write('accuracy\tcoarse_category\tfine_category\n')

            for l in lines:
                l[1] = int(l[1])
                if l[1] in recorded_events.keys():
                    if l[0] in category_mapper.keys():
                        coarse = category_mapper[l[0]][0]
                        fine = category_mapper[l[0]][1]
                    else:
                        coarse = trig_to_cats[l[1]][0]
                        fine = trig_to_cats[l[1]][1]
                    ### proposed by BIDS
                    o.write('{}\t0.750\t{}\t{}\t{}\t'.format(
                            recorded_events[l[1]], l[0], l[1], l[5].strip()))
                    ### Dataset-specific
                    acc = 1 if l[3] == l[4] else 0
                    o.write('{}\t{}\t{}\n'.format(acc, coarse, fine))

        ### Checking all events correspond
        ordered_events = collector[events_out.replace('events.tsv', 'eeg.set')][1]
        with open(events_out) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        header = lines[0]
        vals = lines[1:]
        marker = False
        for l, e in zip(vals, ordered_events):
            if int(l[3]) != e:
                marker = True
        if marker:
            print('corrected recorded events for : {}'.format(s_f, f))
            for l, e in zip(vals, ordered_events):
                l[2] = trig_to_names[e]
                l[3] = str(int(e))
                l[4] = 'na'
                l[5] = 'na'
                l[6] = trig_to_cats[e][0]
                l[7] = trig_to_cats[e][1]
            with open(events_out, 'w') as o:
                o.write('{}\n'.format('\t'.join(header)))
                for l in vals:
                    o.write('{}\n'.format('\t'.join(l)))
        ### re-checking
        with open(events_out) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        header = lines[0]
        vals = lines[1:]
        marker = False
        for l, e in zip(vals, ordered_events):
            if int(l[3]) != e:
                marker = True
        assert marker == False

        if args.write_rawdata:
            os.system('cp {} {}'.format(events_out, os.path.join(s_raw, out_file)))
