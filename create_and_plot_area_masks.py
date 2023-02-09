import matplotlib
import nilearn
import numpy
import os

from matplotlib import pyplot
from nilearn import datasets, plotting, surface
from tqdm import tqdm

from atlas_harvard_oxford_lobes import harvard_oxford_to_lobes

fsaverage = datasets.fetch_surf_fsaverage()
dataset = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
#dataset = nilearn.datasets.fetch_coords_dosenbach_2010()
maps = dataset['maps']
maps_data = maps.get_fdata()
labels = dataset['labels']
collector = {l : numpy.array([], dtype=numpy.float64) for l in labels}

lobes = harvard_oxford_to_lobes()
areas = [
        ### ATLs
        'anterior_temporal_lobe',
        ### Lobes
        'frontal_lobe',
        'temporal_lobe',
        'parietal_lobe',
        'occipital_lobe',
        'limbic_system',
        ]
networks = [
        ### Networks
        'language_network', 
        'general_semantics_network',
        'default_mode_network', 
        'social_network', 
        ]

out = 'fmri_masks'

for area in tqdm(areas):

    relevant_labels = [i for i, l in enumerate(labels) if l in lobes[area]]

    for side in ['left', 'right', 'bilateral']:

        side_area = '{}_{}'.format(side, area)

        if side == 'bilateral':
            msk = numpy.array([1. if v in relevant_labels else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
        elif side == 'right':
            msk = numpy.zeros(maps_data.shape)
            for x in range(maps_data.shape[0]):
                for y in range(maps_data.shape[1]):
                    for z in range(maps_data.shape[2]):
                        if x > maps_data.shape[0]/2:
                            if maps_data[x, y, z] in relevant_labels:
                                msk[x, y, z] = 1.
        elif side == 'left':
            msk = numpy.zeros(maps_data.shape)
            for x in range(maps_data.shape[0]):
                for y in range(maps_data.shape[1]):
                    for z in range(maps_data.shape[2]):
                        if x < maps_data.shape[0]/2:
                            if maps_data[x, y, z] in relevant_labels:
                                msk[x, y, z] = 1.
        print(side_area)
        print(sum(msk.flatten()))
        atl_img = nilearn.image.new_img_like(maps, msk)
        atl_img.to_filename(os.path.join('fmri_masks', '{}.nii.gz'.format(side_area)))
        ### Right
        texture = surface.vol_to_surf(atl_img, fsaverage.pial_right,
                                      interpolation='nearest',
                )
        r = plotting.plot_surf_stat_map(
                    fsaverage.pial_right, 
                    texture, 
                    hemi='right',
                    title='{} - right hemisphere'.format(side_area), 
                    threshold=0.05, 
                    #view='dorsal',
                    #bg_map=fsaverage.sulc_right,
                    bg_on_map=False,
                    bg_map=None,
                    darkness=0.5,
                    cmap='spring', 
                    #cmap=cmaps[side],
                    #view='medial',
                    alpha=0.1,
                    #cmap='Spectral_R'
                    )
        '''
        r.savefig(os.path.join(out, \
                    'surface_right_{}.svg'.format(side),
                    ),
                    dpi=600
                    )
        '''
        r.savefig(os.path.join(out, \
                    'surface_right_{}.jpg'.format(side_area),
                    ),
                    dpi=600
                    )
        pyplot.clf()
        pyplot.close()
        ### Left
        texture = surface.vol_to_surf(atl_img, 
                                      fsaverage.pial_left,
                                      interpolation='nearest',
                )
        l = plotting.plot_surf_stat_map(
                    fsaverage.pial_left, 
                    texture, 
                    hemi='left',
                    title='{} - left hemisphere'.format(side_area), 
                    #view='dorsal',
                    colorbar=True,
                    threshold=0.05, 
                    #bg_map=fsaverage.sulc_left,
                    bg_on_map=False,
                    bg_map=None,
                    cmap='spring', 
                    #cmap='Wistia',
                    #cmap=cmaps[side],
                    #view='medial',
                    darkness=0.5,
                    alpha=0.1,
                    )
        '''
        l.savefig(os.path.join(out, \
                    'surface_left_{}.svg'.format(side),
                    ),
                    dpi=600
                    )
        '''
        l.savefig(os.path.join(out, \
                    'surface_left_{}.jpg'.format(side_area),
                    ),
                    dpi=600
                    )
        pyplot.clf()
        pyplot.close()
'''
### ATLs: left, right and bilateral


### using fedorenko parcels

map_path = os.path.join('region_maps', 'maps','allParcels_language_SN220.nii')
assert os.path.exists(map_path)
print('Masking Fedorenko lab\'s language areas...')
maps = nilearn.image.load_img(map_path)
maps_data = maps.get_fdata()

for side in ['bilateral', 'left', 'right']:
    ### label numbers for temporal lobes
    relevant_labels = [i for i, l in enumerate(labels) if l in lobes['overall anterior temporal lobe']]
    if side == 'left':
        relevant_labels = [4.]
    if side == 'right':
        relevant_labels = [10.]
    if side == 'bilateral':
        relevant_labels = [4., 10.]
    msk = numpy.array([1. if v in relevant_labels else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
    if side == 'bilateral':
        msk = numpy.array([1. if v in relevant_labels else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
    elif side == 'right':
        msk = numpy.zeros(maps_data.shape)
        for x in range(maps_data.shape[0]):
            for y in range(maps_data.shape[1]):
                for z in range(maps_data.shape[2]):
                    if x >= maps_data.shape[1]/2:
                        if maps_data[x, y, z] in relevant_labels:
                            msk[x, y, z] = 1.
    elif side == 'left':
        msk = numpy.zeros(maps_data.shape)
        for x in range(maps_data.shape[0]):
            for y in range(maps_data.shape[1]):
                for z in range(maps_data.shape[2]):
                    if x < maps_data.shape[1]/2:
                        if maps_data[x, y, z] in relevant_labels:
                            msk[x, y, z] = 1.
    print(side)
    print(sum(msk.flatten()))
    atl_img = nilearn.image.new_img_like(maps, msk)
    atl_img.to_filename(os.path.join('region_maps', 'maps', '{}_atl.nii.gz'.format(side)))
    out = os.path.join('region_maps', 'plots')
    ### Right
    texture = surface.vol_to_surf(atl_img, fsaverage.pial_right,
                                  interpolation='nearest',
            )
    r = plotting.plot_surf_stat_map(
                fsaverage.pial_right, texture, hemi='right',
                title='{} ATL - right hemisphere'.format(side), colorbar=True,
                threshold=.05, 
                #bg_map=fsaverage.sulc_right,
                bg_on_map=False,
                bg_map=None,
                darkness=0.5,
                cmap=cmaps[side],
                #view='medial',
                alpha=0.05,
                #cmap='Spectral_R'
                )
    r.savefig(os.path.join(out, \
                '{}_surface_right_atl.svg'.format(side),
                ),
                dpi=600
                )
    r.savefig(os.path.join(out, \
                '{}_surface_right_atl.jpg'.format(side),
                ),
                dpi=600
                )
    pyplot.clf()
    pyplot.close()
    ### Left
    texture = surface.vol_to_surf(atl_img, fsaverage.pial_left,
                                  interpolation='nearest',
            )
    l = plotting.plot_surf_stat_map(
                fsaverage.pial_left, 
                texture, 
                hemi='left',
                title='{} ATL - left hemisphere'.format(side), colorbar=True,
                threshold=.05, 
                #bg_map=fsaverage.sulc_left,
                bg_on_map=False,
                bg_map=None,
                #cmap='Spectral_R', 
                #cmap='Wistia',
                cmap=cmaps[side],
                #view='medial',
                darkness=0.5,
                alpha=0.05,
                )
    l.savefig(os.path.join(out, \
                '{}_surface_left_atl.svg'.format(side),
                ),
                dpi=600
                )
    l.savefig(os.path.join(out, \
                '{}_surface_left_atl.jpg'.format(side),
                ),
                dpi=600
                )
    pyplot.clf()
    pyplot.close()

### pSTS and IFG (both left)

for area in ['left_pSTS', 'left_IFG', 'left_MFG',
             'right_pSTS', 'right_IFG', 'right_MFG']:
    ### left
    if area == 'left_pSTS':
        relevant_labels = [5., 6.]
    elif area == 'left_IFG':
        relevant_labels = [1., 2.]
    elif area == 'left_MFG':
        relevant_labels = [3.]
    ### right
    if area == 'right_pSTS':
        relevant_labels = [11., 12.]
    elif area == 'right_IFG':
        relevant_labels = [7., 8.]
    elif area == 'right_MFG':
        relevant_labels = [9.]
    ### label numbers for relevant area
    #relevant_labels = [i for i, l in enumerate(labels) if l in lobes[area]]
    #print(relevant_labels)
    msk = numpy.array([1. if v in relevant_labels else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
    msk = numpy.zeros(maps_data.shape)
    for x in range(maps_data.shape[0]):
        for y in range(maps_data.shape[1]):
            for z in range(maps_data.shape[2]):
                if x >= maps_data.shape[1]/2:
                    if maps_data[x, y, z] in relevant_labels:
                        msk[x, y, z] = 1.
    print(area)
    print(sum(msk.flatten()))
    atl_img = nilearn.image.new_img_like(maps, msk)
    out = os.path.join('region_maps', 'plots')
    if 'left' in area:
        surf_mesh=fsaverage.pial_left
        hemi='left'
        atl_img.to_filename(os.path.join('region_maps', 'maps', '{}.nii.gz'.format(area)))
    elif 'right' in area:
        surf_mesh=fsaverage.pial_right 
        hemi='right'
    ### Right
    texture = surface.vol_to_surf(
                                  atl_img, 
                                  surf_mesh=surf_mesh,
                                  interpolation='nearest',
                                  )
    r = plotting.plot_surf_stat_map(
                        surf_mesh=surf_mesh, 
                        stat_map=texture, 
                        hemi=hemi,
                        title=area.replace('_', ' '), 
                        colorbar=True,
                        threshold=.05, 
                        #bg_map=fsaverage.sulc_right,
                        bg_map=None,
                        bg_on_map=False,
                        darkness=0.5,
                        cmap=cmaps[area.split('_')[-1]],
                        #view='medial',
                        alpha=0.05,
                        #cmap='Spectral_R'
                        )
    r.savefig(os.path.join(out, \
                'surface_{}.svg'.format(area),
                ),
                dpi=600
                )
    r.savefig(os.path.join(out, \
                'surface_{}.jpg'.format(area),
                ),
                dpi=600
                )
    pyplot.clf()
    pyplot.close()

for network in ['vmpfc', 'general_semantics', 'fedorenko_language', 'control_semantics']: 
    if network == 'general_semantics':
        map_path = os.path.join('region_maps', 'maps', 'General_semantic_cognition_ALE_result.nii')
        assert os.path.exists(map_path)
        print('Masking general semantics areas...')
        map_nifti = nilearn.image.load_img(map_path)
        map_nifti = nilearn.image.binarize_img(map_nifti, threshold=0.)
        #map_nifti = nilearn.image.resample_to_img(map_nifti, single_run, interpolation='nearest')
    elif network == 'control_semantics':
        map_path = os.path.join('region_maps', 'maps', 'semantic_control_ALE_result.nii')
        assert os.path.exists(map_path)
        print('Masking control semantics areas...')
        map_nifti = nilearn.image.load_img(map_path)
        map_nifti = nilearn.image.binarize_img(map_nifti, threshold=0.)
        #map_nifti = nilearn.image.resample_to_img(map_nifti, single_run, interpolation='nearest')
    elif network == 'fedorenko_language':
        map_path = os.path.join('region_maps', 'maps', 'allParcels_language_SN220.nii')
        #map_path = os.path.join('region_maps', 'maps', 'langloc_n806_p<0.05_atlas.nii')
        assert os.path.exists(map_path)
        print('Masking Fedorenko lab\'s language areas...')
        map_nifti = nilearn.image.load_img(map_path)
        #map_nifti = nilearn.image.binarize_img(map_nifti, threshold=0.)
    if network == 'vmpfc':
        map_path = os.path.join('region_maps', 'maps', 'VMPFC-mask-final.nii.gz')
        assert os.path.exists(map_path)
        print('Masking VMPFC...')
        map_nifti = nilearn.image.load_img(map_path)
        map_nifti = nilearn.image.binarize_img(map_nifti, threshold=0.)
    print(network)
    print(sum(map_nifti.get_fdata().flatten()))
    ### Right
    texture = surface.vol_to_surf(map_nifti, fsaverage.pial_right, interpolation='nearest')
    r = plotting.plot_surf_stat_map(
                fsaverage.pial_right, texture, hemi='right',
                title='{} - right hemisphere'.format(network.replace('_', ' ')), colorbar=True,
                threshold=.05, 
                bg_map=fsaverage.sulc_right,
                bg_on_map=False,
                darkness=0.5,
                cmap=cmaps[network],
                #view='medial',
                alpha=0.3,
                #cmap='Spectral_R'
                )
    r.savefig(os.path.join(out, \
                'surface_right_{}.svg'.format(network),
                ),
                dpi=600
                )
    r.savefig(os.path.join(out, \
                'surface_right_{}.jpg'.format(network),
                ),
                dpi=600
                )
    pyplot.clf()
    pyplot.close()
    ### Left
    texture = surface.vol_to_surf(map_nifti, fsaverage.pial_left)
    l = plotting.plot_surf_stat_map(
                fsaverage.pial_left, texture, hemi='left',
                title='{} - left hemisphere'.format(network.replace('_', ' ')), colorbar=True,
                threshold=.05, 
                bg_map=fsaverage.sulc_left,
                bg_on_map=False,
                #cmap='Spectral_R', 
                #cmap='Wistia',
                cmap=cmaps[network],
                #view='medial',
                darkness=0.5,
                alpha=0.3,
                )
    l.savefig(os.path.join(out, \
                'surface_left_{}.svg'.format(network),
                ),
                dpi=600
                )
    l.savefig(os.path.join(out, \
                'surface_left_{}.jpg'.format(network),
                ),
                dpi=600
                )
    pyplot.clf()
    pyplot.close()
'''
