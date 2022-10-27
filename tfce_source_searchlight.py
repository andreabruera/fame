import argparse
import logging
import mne
import nilearn
import numpy
import os
import scipy

from nilearn import datasets, image, plotting
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from scipy import stats
from tqdm import tqdm

from io_utils import ExperimentInfo, prepare_folder

parser = argparse.ArgumentParser()
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

args.data_folder = '../../dataset/neuroscience/family_lexicon_eeg/derivatives'
args.data_kind = 'erp'
args.entities = 'individuals_only'
args.average = 24
args.corrected = True
args.subsample = 'subsample_8'
args.wv_dim_reduction = 'no_dim_reduction'

nii_folder = prepare_folder(args)
print(nii_folder)
out_folder = nii_folder.replace('results', 'plots')
os.makedirs(out_folder, exist_ok=True)

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
numpy.seterr(all='raise')

template = nilearn.datasets.load_mni152_template(resolution=6)

data = list()

print('Loading images')
with tqdm() as pbar:
    nii_files = os.listdir(nii_folder)
    #assert len(nii_files) == 33
    for f in nii_files:
        nii_img = nilearn.image.load_img(os.path.join(nii_folder, f))
        mask = nilearn.masking.compute_brain_mask(nii_img)
        sub_data = nilearn.masking.apply_mask(nii_img, mask)
        print(numpy.amin(sub_data))
        print(numpy.amax(sub_data))
        data.append(sub_data)
        pbar.update(1)
data = numpy.stack(data)

print('Loading adjacency matrix')
### Computing adjacency matrix
whole_brain = nilearn.masking.compute_brain_mask(nii_img)
### World coordinates of the seeds
bool_mask = whole_brain.get_fdata().astype(bool)
process_mask_coords = numpy.where(whole_brain.get_fdata()!=0)
process_mask_coords = nilearn.image.resampling.coord_transform(
                    process_mask_coords[0], process_mask_coords[1],
                    process_mask_coords[2], whole_brain.affine)
process_mask_coords = numpy.asarray(process_mask_coords).T
_, adj_matrix = _apply_mask_and_get_affinity(
                                             process_mask_coords, 
                                             nii_img, 
                                             radius=12., 
                                             allow_overlap=True, 
                                             mask_img=whole_brain
                                             )
full_adj_matrix = adj_matrix.copy()
### Considering time contiguity
#full_adj_matrix = mne.stats.combine_adjacency(17, adj_matrix)
### Not considering it
'''
### row
for r in range(12):
    ### column
    ### initialize
    row_adj_matrix = scipy.sparse.lil_matrix((adj_matrix.shape[0], 0))
    for c in range(12):
        ### diagonal
        if r == c:
            ### add
            row_adj_matrix = scipy.sparse.hstack((row_adj_matrix, adj_matrix))
        ### add zeros
        else:
            row_adj_matrix = scipy.sparse.hstack(
                                                 (
                                                  row_adj_matrix, 
                                                  scipy.sparse.lil_matrix((adj_matrix.shape[0], adj_matrix.shape[0]))
                                                  )
                                                 )
    if r == 0:
        full_adj_matrix = row_adj_matrix.copy()
    else:
        full_adj_matrix = scipy.sparse.vstack((full_adj_matrix, row_adj_matrix))
    del row_adj_matrix
'''

print('Loaded!')
empty_brain = numpy.zeros(whole_brain.shape)
results = list()

data = data - 0.5
#t_data = data.copy()
for t in range(data.shape[1]):
    t_data = data[:, t, :]
    ts,one,ps,two = mne.stats.permutation_cluster_1samp_test(
                                                       t_data, 
                                                       tail=1,
                                                       threshold=dict(start=0, step=0.2),
                                                       #adjacency=None,
                                                       adjacency=full_adj_matrix, 
                                                       n_jobs=int(os.cpu_count()/3),
                                                       #n_permutations=10000,
                                                       )
    print([t, numpy.amin(ps)])
    current_brain = empty_brain.copy()
    current_brain[bool_mask] = 1-ps
    results.append(current_brain)
results = numpy.stack(results)
results = numpy.stack(results, -1)

current_brain = nilearn.image.new_img_like(ref_niimg=whole_brain, 
                                       data=results)
current_brain.to_filename(os.path.join(out_folder, 'all_subjects.nii.gz'))

### Plotting
t_zero = -70
threshold = 0.95
for t in range(data.shape[1]):
    idx_img = nilearn.image.index_img(current_brain, t)
    t_zero += 70
    end_t = t_zero + 70
    if numpy.amax(idx_img.get_fdata()) >= threshold:
        title = 'Time: {}-{} ms, {}, {}, threshold={}'.format(
                                                t_zero, 
                                                end_t, 
                                                args.analysis,
                                                args.semantic_category,
                                                threshold
                                                )
        output_file = os.path.join(out_folder,
                                   'time_{}-{}_threshold-{}.jpg'.format(t_zero, 
                                                           end_t,
                                                           threshold)
                                   )
        nilearn.plotting.plot_glass_brain(
                                          idx_img,
                                          title=title,
                                          threshold=threshold,
                                          output_file=output_file
                                          )

        ### Finding out places where results are significant
        dataset = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
        maps = dataset['maps']
        maps_data = maps.get_fdata()
        labels = dataset['labels']
        interpr_nifti = nilearn.image.resample_to_img(idx_img, maps, interpolation='nearest')
        collector = {l : numpy.array([], dtype=numpy.float64) for l in labels}
        for l_i, label in enumerate(labels):
            if l_i > 0:
                msk = maps_data == l_i
                mskd = nilearn.masking.apply_mask(interpr_nifti, nilearn.image.new_img_like(maps, msk))
                collector[label] = mskd[mskd>=threshold]
        collector = sorted({k : v.shape[0] for k, v in collector.items()}.items(), key=lambda items : items[1],
                            reverse=True)
        total = sum(list([k[1] for k in collector]))
        percentages = {k[0] : k[1]/total for k in collector if k[1] != 0.}
        output_perc = output_file.replace('jpg', 'txt')
        with open(output_perc, 'w') as o:
            for k, v in percentages.items():
                o.write('{}\t{}%\n'.format(k, round(v*100, 2)))
