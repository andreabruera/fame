import argparse
import nilearn
import nibabel
import numpy
import os

import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage
from nilearn import datasets, image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', choices=['one', 'two'],
                    required=True)
args = parser.parse_args()

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
srf_src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
vol_src = os.path.join(fs_dir, 'bem', 'fsaverage-vol-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

montage = mne.channels.make_standard_montage('biosemi128')

if args.experiment_id == 'two':
    folders = ['family_lexicon_eeg']
else:
    folders = ['exploring_individual_entities_eeg']
for folder in folders:
    for s in tqdm(range(1, 34)):

        collector = dict()
        out_f = os.path.join('/', 'import', 'cogsci', 'andrea',
                             'dataset', 'neuroscience', 
                             folder, 'reconstructed', 'sub-{:02}'.format(s),)
        os.makedirs(out_f, exist_ok=True)

        ### Loading epochs
        f = os.path.join('/', 'import', 'cogsci','andrea', 
                         'dataset', 'neuroscience', 
                         folder, 'derivatives', 'sub-{:02}'.format(s), 
                         'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(s))
        epochs = mne.read_epochs(f, preload=True, )
        epochs = epochs.pick_types(eeg=True, stim=False)
        epochs.set_montage(montage)
        epochs.set_eeg_reference(projection=True,)
        ### Cropping time
        #epochs.crop(tmin=0.)
        ### We do not decimate, but average responses later
        #epochs.decimate(8)
        avg_epochs = list()
        all_evoked = epochs.average(by_event_type=True)
        for evoked in all_evoked:
            avg_evoked = list()
            data = evoked.get_data()
            t_points = numpy.array((-0.1, 0.))
            for i in range(11):
                t_indices = [t_i for t_i, t in enumerate(evoked.times) if t>=t_points[0] and t<t_points[1]]
                print(t_indices)
                t_point = numpy.average(data[:, t_indices], axis=1)
                assert t_point.shape == (128,)
                avg_evoked.append(t_point)
                t_points += 0.1
            avg_evoked = numpy.stack(avg_evoked, axis=-1)
            assert avg_evoked.shape == (128, 11)
            avg_evoked = mne.EvokedArray(
                                         data=avg_evoked,
                                         tmin=evoked.tmin,
                                         kind=evoked.kind,
                                         baseline=None,
                                         info=evoked.info, 
                                         comment=evoked.comment, 
                                         nave=evoked.nave
                                         )

            avg_epochs.append(avg_evoked)

        ### Computing forward solution
        ### Surface
        #fwd = mne.make_forward_solution(epochs.info, trans=trans, src=srf_src, 
        #                                bem=bem, eeg=True, meg=False,
        #                                mindist=5.0, n_jobs=os.cpu_count())
        #f_out = f.replace('epo.fif', 'surface-fwd.fif')
        #mne.write_forward_solution(f_out, fwd, overwrite=True)
        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=vol_src, 
                                        bem=bem, eeg=True, meg=False,
                                        mindist=5.0, n_jobs=os.cpu_count())
        #f_out = f.replace('epo.fif', 'volume-fwd.fif')
        #mne.write_forward_solution(f_out, fwd, overwrite=True)

        ### Computing the covariance matrix
        #f_out = f.replace('epo.fif', '-cov.fif')
        noise_cov_baseline = mne.compute_covariance(epochs, tmax=0, method='auto', rank=None)
        #mne.write_cov(f_out, noise_cov_baseline, overwrite=True)
        inverse_operator = mne.minimum_norm.make_inverse_operator(
                                                 epochs.info, fwd, noise_cov_baseline,
                                                 )
        del fwd
        method = "eLORETA"
        snr = 3.
        lambda2 = 1. / snr ** 2
        for evoked in avg_epochs:
            trigger = int(evoked.comment)
            stc, residual = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2,
                                          method=method, pick_ori=None,
                                          return_residual=True, verbose=True)
            fmri_image = stc.as_volume(mne.read_source_spaces(vol_src))
            mni_img = nilearn.image.resample_to_img(fmri_image, nilearn.datasets.load_mni152_template(resolution=6))
            out_file = os.path.join(out_f, 'trigger_{}.nii.gz'.format(trigger))
            nibabel.loadsave.save(mni_img, out_file)
            
            #raise RuntimeError('Script correctly running!')
