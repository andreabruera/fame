import mne
import numpy
import os
import sys

from matplotlib import pyplot
from scipy import stats

sys.path.append('/import/cogsci/andrea/github/fame')
from searchlight.searchlight_utils import SearchlightClusters

overall = dict()
for typ in ['coarse', 'fine']:

    folder = 'results/one/erp/searchlight_whole_trial_classification_{}/individuals_only/subsample/average_0/people_and_places/'.format(typ)

    subjects = list()

    for i in range(1, 33+1):
        sub_id = 'sub_{:02}'.format(i)
        electrodes = list()
        for e in range(128):
            f_name = '{}_cluster_{}_accuracy_scores.txt'.format(sub_id, e)
            with open(os.path.join(folder, f_name)) as i:
                lines = float([l.strip() for l in i.readlines()][-1])
            electrodes.append(lines)
        subjects.append(electrodes)

    subjects = numpy.array(subjects)
    overall[typ] = subjects

c = numpy.average(overall['coarse'], axis=0)
f = numpy.average(overall['fine'], axis=0)
corr = stats.spearmanr(c, f)

sorted_c = sorted([(c_i, co) for c_i, co in enumerate(c)], key=lambda item : item[1], reverse=True)
sorted_f = sorted([(f_i, fi) for f_i, fi in enumerate(f)], key=lambda item : item[1], reverse=True)

with open('classification_scores_per_electrode.txt', 'w') as o:
    o.write('electrode index\tcoarse\tfine\n')
    for i in range(128):
        o.write('{}\t{}\t{}\n'.format(i, c[i], f[i]))

### Plotting on the scalp

info = mne.create_info(ch_names=[v for k, v in SearchlightClusters().index_to_code.items()], \
                       #the step is 8 samples, so we divide the original one by 7
                       sfreq=256/8, \
                       ch_types='eeg')
montage = mne.channels.make_standard_montage('biosemi128')


c = numpy.array([[co, co, co] for co in c])
f = numpy.array([[fi, fi, fi] for fi in f])
for res, res_name in zip([c, f], ['coarse', 'fine']):

    cmap = 'plasma' 

    title = 'Electrode accuracy performance for {}'.format(res_name)
    evoked = mne.EvokedArray(res, info=info, tmin=0.)
    evoked.set_montage(montage)

    evoked.plot_topomap(ch_type='eeg', time_unit='s', times=[0.],
                        units='accuracy', scalings={'eeg':1.}, \
                        vmin=min(res[:, 0]), vmax=max(res[:, 0]),
                        cmap=cmap, title=title)

    pyplot.savefig(os.path.join('plots', 'prova_{}.jpg'.format(res_name)), dpi=300)
    pyplot.clf()
