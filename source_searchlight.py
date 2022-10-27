import argparse
import itertools
import logging
import nibabel
import nilearn
import numpy
import os
import random
import re
import scipy
import sklearn

from nibabel import loadsave
from nilearn import decoding, image, masking
from scipy import stats
from sklearn import linear_model
from tqdm import tqdm

from io_utils import ExperimentInfo, prepare_folder

def load_reconstructed_epochs(args, sub):
    s_folder = os.path.join(args.nii_folder, 'sub-{:02}'.format(sub))
    files = [f for f in os.listdir(s_folder) if 'nii.gz' in f]
    n = 40 if args.experiment_id == 'one' else 32
    assert len(files) == 32
    collector =dict()
    for f in files:
        img = nilearn.image.load_img(os.path.join(s_folder, f))
        label = re.sub('^.+_|[.].+$', '', f)
        collector[int(label)] = img.get_fdata()
    return collector, img

parser = argparse.ArgumentParser()
parser.add_argument('--nii_folder', required=True, type=str,
                    help='Where are the nii files?')
parser.add_argument('--experiment_id', \
                    choices=['one', 'two', 'pilot'], \
                    required=True, help='Indicates which \
                    experiment to run')
parser.add_argument('--analysis', \
                    choices=['coarse_source_searchlight', 
                        'famous_familiar_source_searchlight', 
                        'fine_source_searchlight'], 
                    required=True, help='Indicates which '\
                    'experiment to run')
parser.add_argument('--semantic_category', choices=['people', 'places', 
                                                    'famous', 'familiar',
                                                    'all',
                                                    ], \
                    required=True, help='Which semantic category?') 
args = parser.parse_args()

args.data_folder = args.nii_folder.replace('reconstructed/', '')
args.data_kind = 'erp'
args.entities = 'individuals_only'
args.average = 24
args.corrected = True
args.subsample = 'subsample_8'
args.wv_dim_reduction = 'no_dim_reduction'

out_folder = prepare_folder(args)
print(out_folder)
os.makedirs(out_folder, exist_ok=True)

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
numpy.seterr(all='raise')

cat_index = 1 if 'coarse' in args.analysis else 2

### Running Searchlight for each subject
for s in range(1, 34):
    experiment = ExperimentInfo(args, subject=s)

    ### Reading files
    epochs, nii_img = load_reconstructed_epochs(args, s)
    ### Reducing epochs to actually present stimuli
    epochs = {k : v for k, v in epochs.items() if k in experiment.trigger_to_info.keys()}
    time_points = list(set([v.shape[-1] for v in epochs.values()]))[0]

    ### Iterating through time points
    t_scores = list()
    for time_point in tqdm(range(time_points)):
        ### Reading the current time point
        current_samples = {k : v[:,:,:,time_point] for k, v in epochs.items()}
        ### Checking shape is fine
        for v in current_samples.values():
            assert v.shape[:3] == nii_img.shape[:3]
        ### Distinguishing samples and triggers
        all_samples = [v for k, v in current_samples.items()]
        all_labels = [k for k, v in current_samples.items()]
        ### Creating the actual train/test splits from triggers
        cv = [([all_labels.index(k) for k in experiment.trigger_to_info.keys() if k not in test], [all_labels.index(t) for t in test]) for test in experiment.test_splits]
        ### Turning triggers to labels
        all_labels = [experiment.trigger_to_info[k][cat_index] for k in all_labels]
        ### Stacking all responses
        current_samples = numpy.stack(tuple(all_samples), axis=-1)
        ### Creating 4D image from samples
        img_4d = nilearn.image.new_img_like(ref_niimg=nii_img,
                      data=current_samples)
        whole_brain = nilearn.masking.compute_brain_mask(img_4d)
        estimator = sklearn.linear_model.RidgeClassifier()
        searchlight = nilearn.decoding.SearchLight(
                        whole_brain,
                        process_mask_img=whole_brain, 
                        estimator=estimator,
                        radius=20., 
                        n_jobs=int(os.cpu_count()/2), 
                        verbose=0, 
                        cv=cv
                        )
        searchlight.fit(img_4d, all_labels)
        t_scores.append(searchlight.scores_)
    ### Stacking classification scores
    t_scores = numpy.stack(tuple(t_scores), axis=-1)
    ### Checking shape is O.K.
    assert t_scores.shape[:3] == nii_img.shape[:3]
    ### Writing to file the classification performance map
    out_img = nilearn.image.new_img_like(
                               ref_niimg=nii_img,
                               data=t_scores)
    nibabel.loadsave.save(out_img, 
                          os.path.join(out_folder, 
                                       'sub-{:02}_source-searchlight.nii.gz'.format(s)))
