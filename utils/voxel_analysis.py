from sklearn.cluster import DBSCAN
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression, VarianceThreshold

def voxel_analysis(args, collection, masker, aal_atlas, talairach_atlas, model_path, constant_features_remover = [], conditions = []):

    constant_features_remover = VarianceThreshold(threshold=0)

    selected_voxels_collection = []
    selected_voxels = numpy.zeros(len(train_data[0][0]))
    coefficients = m.coef_

    ### WORSE - Crude feature selection version 2: summing all coefficients across dimensions, retain only the top 5%

    summed_coeff = numpy.zeros(len(train_data[0][0]))
    selected_voxels = numpy.zeros(len(train_data[0][0]))
    for dimension in coefficients:
        summed_coeff = numpy.add(summed_coeff, numpy.absolute(dimension))
    percentile_value = numpy.percentile(summed_coeff, 90)
    for voxel_index, individual_coeff in enumerate(summed_coeff):
        if individual_coeff > percentile_value:
            selected_voxels[voxel_index] = individual_coeff
            #selection_mask[voxel_index] = float(1)
        else:
            selected_voxels[voxel_index] = 0.0
            #selection_mask[voxel_index] = float(0)

    selected_voxels_collection.append(selected_voxels)
    # Calculating the mean activations for the voxel which are used consistently in at least 95% of the classifications

    overlap = numpy.zeros(len(collection[0]))
    threshold = round(len(collection) * 0.95)
    for i in range(len(collection[0])):
        counter = 0
        for coef in collection:
            if coef[i] == 0.0:
                pass
            else:
                counter += 1
        if counter >= threshold:
            overlap[i] = float(numpy.mean([k[i] for k in collection]))
        else:
            overlap[i] = 0.0

    if 'Ridge' not in model_path:
    # Putting the voxels back in the mni template space

        features = [k for k in constant_features_remover.get_support()]
        feature_counter = 0
        feature_mapper = defaultdict(int)
        for i, f in enumerate(features):
            if f == True:
                feature_mapper[i] = feature_counter
                feature_counter += 1

        overlapped_and_mni = numpy.zeros(len(features))
        for feature_index, tf in enumerate(features):
            if tf != False:
                overlap_index = feature_mapper[feature_index]
                overlapped_and_mni[feature_index] = overlap[overlap_index]

    else:
        overlapped_and_mni = overlap.copy()

    overlap_mask = masker.inverse_transform(overlapped_and_mni)
    mean_collection = image.mean_img(overlap_mask)

    # Plotting the selected features/voxels

    if args.write_to_file:

        stat_map = plot_stat_map(mean_collection, mni_template, title='{} vs {}'.format(conditions[0], conditions[1]))
        stat_map.savefig(os.path.join(model_path, 'weights_{}_{}.png'.format(conditions[0], conditions[1])))
        stat_map.close()

    # This coming part is for labelling the selected features/voxels

    areas_array = mean_collection.get_fdata()
    voxels = []
    for x in range(areas_array.shape[0]):
        for y in range(areas_array.shape[1]):
            for z in range(areas_array.shape[2]):
                if areas_array[x][y][z] != 0.0:
                    voxels.append([areas_array[x][y][z], [x,y,z]])
    

    # The coming section clusters the voxels before labeling them

    ''' 
    cluster_names = defaultdict(lambda : defaultdict(int))
    clusters = DBSCAN(eps = 1).fit([k[1] for k in voxels])
    cluster_labels = [l for l in clusters.labels_]
    clusters = defaultdict(list)
    for i, l in enumerate(cluster_labels):
        try:
            clusters[l].append(voxels[i])
        except IndexError:
            pass
    for i, c in clusters.items():
        for v in c:
            l = read_atlas_peak(atlas[0], v[1])
            if l != 'no_label':
                cluster_names[i][l] += 1
        try:
            name = [k for k in sorted(cluster_names[i], key = cluster_names[i].get, reverse = True)][0]
            cluster_activation = max([float(v[0]) for v in c])
            cluster_names['{}__{}'.format(i, name)] = cluster_activation
        except IndexError:
            pass
    '''

    voxel_activations = defaultdict(float)
    name_coords = defaultdict(list)
    for i, v in enumerate(voxels):
        l = read_atlas_peak(aal_atlas, v[1])
        if l != 'no_label':
            name = '{}__{}'.format(i, l)
            voxel_activations[name] = abs(v[0])
            name_coords[name] = v[1]
            

    ordered_voxels = {v[0] : v[1] for v in sorted(voxel_activations.items(), key = lambda item : abs(item[1]), reverse = True)}
    final_voxels = defaultdict(list)
    for c in ordered_voxels.items():
        area = re.sub('\d+_',  '', c[0])
        if area not in final_voxels.keys():
            coords = name_coords[c[0]]
            talairach_coords = read_atlas_peak(talairach_atlas, coords)
            final_voxels[area] = [talairach_coords, coords, c[1]]


    if args.write_to_file:
        with open(os.path.join(model_path, 'cluster_coords.txt'), 'a') as o:
            if 'Ridge' not in model_path:
                o.write('Classification:\t{} vs {}\n\n'.format(conditions[0], conditions[1]))
            else:
                o.write('Results for the model predicting individual entities vectors')
            counter = 1
            for i, c in final_voxels.items():
                o.write('Area {}\nAAL label:\t{}\nTalairach label:\t{}\nCoordinates:\t{}\nActivation:\t{}\n\n\n'.format(counter, i, c[0], c[1], c[2]))
                counter += 1
    else:
        if 'Ridge' not in model_path:
            print('Classification:\t{} vs {}\n\n'.format(conditions[0], conditions[1]))
        else:
            print('Results for the model predicting individual entities vectors')
        counter = 1
        for i, c in final_voxels.items():
            logging.info('Area {}\nAAL label:\t{}\nTalairach label:\t{}\nCoordinates:\t{}\nActivation:\t{}\n\n\n'.format(counter, i, c[0], c[1], c[2]))
            counter += 1

# Not sure what this code did... look for the NOT_SURE_MARKER in unified_decoding_script.py
'''
### Feature selection - run only on the ranking evaluation because it involves less cross-validations

if args.voxel_analysis:

    # Reducing the voxels used to 50000         

    if 'reduced' in test_key:
        selection_mask = numpy.zeros(len(train_data[0][0]))
        selection_dict = {k : v for k, v in enumerate(selected_voxels)}
        for counter_index, voxel in enumerate(sorted(selection_dict, key = selection_dict.get, reverse = True)):
            if counter_index < 50000:
                selection_mask[voxel] = float(1)
                
        assert len([k for k in selection_mask if k != float(0)]) <= 50000
        test_data_copy = test_data.copy()
        del test_data
        test_data = numpy.multiply(test_data_copy, selection_mask) 
'''
