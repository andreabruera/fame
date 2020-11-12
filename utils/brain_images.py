import numpy
import nilearn
import re
import os
import collections
import sklearn

from nilearn.plotting import plot_stat_map, show
from utils.general_utilities import get_details
from atlasreader.atlasreader import get_atlas, check_atlases, read_atlas_peak, coord_xyz_to_ijk, check_atlas_bounding_box, get_label

from collections import defaultdict

def find_stable_voxels(amount_stable_voxels, brain_images):
    
    output_dict = collections.defaultdict(list)
    mask_dict = collections.defaultdict(list)
    std_container = collections.defaultdict(list)
    for k, v in brain_images.items():

        normalized_vs = [(vec - min(vec)) / (max(vec) - min(vec)) for vec in v]
        diff = []
        for dimension in range(len(normalized_vs[0])):
            diff.append(numpy.std([vec[dimension] for vec in normalized_vs]))
                
        for i, val in enumerate(diff):
            std_container[i].append(val)

    diff_dict = {k : numpy.nanmean(v) for k, v in std_container.items()}
    mask_indices = [k for i, k in enumerate(sorted(diff_dict, key = diff_dict.get)) if i < amount_stable_voxels]

    for k, vecs in brain_images.items():
        for v in vecs:
            output_dict[k].append(numpy.asarray([v[dim] for dim in sorted(mask_indices)]))

    return output_dict, mask_indices

def pca(brain_images, components=0):

    all_images = [v for k, vecs in brain_images.items() for v in vecs] 
    n_components = components if components != 0 else min(len(all_images), 500)
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca_data = pca.fit_transform(all_images)
    reconstruct_pca = collections.defaultdict(str)
    counter = 0
    for k, vecs in brain_images.items():
        for index, v in enumerate(vecs):
            reconstruct_pca[counter] = k
            counter += 1
    output_images = collections.defaultdict(list)
    for index, vec in enumerate(pca_data):
        output_images[reconstruct_pca[index]].append(vec)
    
    return output_images

def get_brain_areas(s, masker, mask_dict):

    aal_atlas = check_atlases('aal')[0]
    talairach_atlas = check_atlases('talairach_gyrus')[0]
    mni_template =  nilearn.datasets.load_mni152_template()

    voxel_dict = numpy.zeros(68049)
    for k, mask in mask_dict.items():
        for mask_index in mask:
            voxel_dict[mask_index] += 1
    voxel_threshold = numpy.percentile([k for k in voxel_dict], 95)
    final_voxels = numpy.zeros(68049)
    for i, k in enumerate(voxel_dict):
        if k > voxel_threshold:
            final_voxels[i] = 0.5
    inverse = masker.inverse_transform(final_voxels)
    stat_map = plot_stat_map(inverse, mni_template)
    stat_map.savefig('sub_{:02}_weights.png'.format(s))
    stat_map.close()
    # This coming part is for labelling the selected features/voxels

    areas_array = inverse.get_fdata()
    voxels = []
    for x in range(areas_array.shape[0]):
        for y in range(areas_array.shape[1]):
            for z in range(areas_array.shape[2]):
                if areas_array[x][y][z] != 0.0:
                    coords = nilearn.image.coord_transform(x, y, z, mni_template.affine)
                    voxels.append([areas_array[x][y][z], coords])

    voxel_activations = collections.defaultdict(float)
    name_coords = collections.defaultdict(list)
    for i, v in enumerate(voxels):
        l = read_atlas_peak(aal_atlas, v[1])
        if l != 'no_label':
            name = '{}_{}'.format(i, re.sub('_', ' ', l))
            voxel_activations[name] = abs(v[0])
            name_coords[name] = v[1]
            

    ordered_voxels = {v[0] : v[1] for v in sorted(voxel_activations.items(), key = lambda item : abs(item[1]), reverse = True)}
    final_voxels = collections.defaultdict(list)
    for c in ordered_voxels.items():
        area = re.sub('\d+_',  '', c[0])
        if area not in final_voxels.keys():
            coords = name_coords[c[0]]
            talairach_coords = read_atlas_peak(talairach_atlas, coords)
            final_voxels[area] = [talairach_coords, coords, c[1]]
    with open('sub_{:02}_cluster_coords.txt'.format(s), 'w') as o:
        for i, c in final_voxels.items():
            o.write('AAL label:\t{}\nTalairach label:\t{}\nCoordinates:\t{}\n\n'.format(i, c[0], c[1]))
