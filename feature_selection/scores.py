import itertools
import math
import numpy

from scipy import stats
from tqdm import tqdm

from word_vector_enc_decoding.read_word_vectors import WordVectors

def get_scores(args, experiment, full_eeg):

    feature_selection = args.feature_selection_method
    original_shape = list(set([vec.shape for k,v in full_eeg.items() for vec in v]))[0]
    flat_data = {k : [vec.flatten() for vec in v] for k, v in full_eeg.items()}
    all_dimensions = flat_data[list(flat_data.keys())[0]][0].shape[0]
    #assert clusterized_array.flatten().shape[0] == all_dimensions
    ## Averaging by groups of 4
    averaged_data = {k : [numpy.average(v[i:i+4], axis=0) for i in range(0, len(v), 4) \
                                       if i+4 < len(v)-1] for k, v in flat_data.items()}
    ### Reducing to minimum common number of ERPs available
    max_vecs = min([len(v) for k, v in averaged_data.items()])
    averaged_data = {k : v[:max_vecs] for k, v in averaged_data.items()}

    '''
    ### Working by leave-two-out
    left_out_combs = list(itertools.combinations(list(full_eeg.keys()), 2))

    scores = {feature_selection : {l_c : list() for l_c in left_out_combs}}

    for l_c in tqdm(left_out_combs):
     
        # Reducing the features
        comb_eeg = {k : v for k, v in averaged_data.items() if k not in l_c}
        assert len(comb_eeg) == len(averaged_data)-2    
    
        # Get scores separately for each method
        scores[feature_selection][l_c] = select(feature_selection, comb_eeg, all_dimensions)

    output_scores = [[k, v] for k,v in scores.items()]   

    return output_scores
    '''
    ### Working with leave-one-subject out
    scores = select(args, experiment, averaged_data, all_dimensions)

    return scores


def select(args, experiment, eeg_data, all_dimensions):

    f = args.feature_selection_method
    if f == 'distinctiveness':
        scores = distinctiveness(eeg_data, all_dimensions)
    elif f == 'noisiness':
        scores = noisiness(eeg_data, all_dimensions)
    elif f == 'stability':
        scores = stability(eeg_data, all_dimensions)
    elif f == 'mutual_information':
        scores = mutual_information(eeg_data, all_dimensions)
    elif f == 'fisher':
        scores = fisher(eeg_data, all_dimensions)
    elif f == 'attribute_correlation':
        scores = attribute_correlation(eeg_data, all_dimensions, args, experiment)

    return scores

def distinctiveness(averaged_data, all_dimensions):

    dist_array = numpy.zeros(all_dimensions, dtype=numpy.double)
    ### Distinctiveness
    #dim_distinctiveness = list()
    for d in tqdm(range(all_dimensions)):
        dimension_list = numpy.array([[vec[d] for vec in v] for k, v in averaged_data.items()]).T
        distinctiveness_score = numpy.average(numpy.std(dimension_list, axis=1))
        dist_array[d] = distinctiveness_score
        #dim_distinctiveness.append((d, distinctiveness_score))
    #sorted_distinctiveness = sorted(dim_distinctiveness, key=lambda item : item[1], reverse=True)
    #distinctiveness_scores[l_c] = sorted_distinctiveness

    #return distinctiveness_scores
    #return sorted_distinctiveness
    return dist_array

def noisiness(averaged_data, all_dimensions):

    ### Noisiness
    #dim_noisiness = list()
    noisiness_array = numpy.zeros(all_dimensions, dtype=numpy.double)
    for d in tqdm(range(all_dimensions)):
        dimension_list = numpy.array([[vec[d] for vec in v] for k, v in averaged_data.items()]).T
        noisiness_score = numpy.average(numpy.std(dimension_list, axis=0))
        #dim_noisiness.append((d, noisiness_score))
        noisiness_array[d] = noisiness_score
    #sorted_noisiness = sorted(dim_noisiness, key=lambda item : item[1])

    #return sorted_noisiness
    return noisiness_array

def fisher(averaged_data, all_dimensions):

    ### Fisher
    fisher_array = numpy.zeros(all_dimensions, dtype=numpy.double)
    for d in tqdm(range(all_dimensions)):
        dimension_list = numpy.array([[vec[d] for vec in v] for k, v in averaged_data.items()]).T
        overall_dimension_mean = numpy.average(dimension_list)
        per_word_dimension_mean = numpy.average(dimension_list, axis=0)
        per_word_dimension_var = numpy.var(dimension_list, axis=0)
        numerator = numpy.sum(dimension_list.shape[1]*(per_word_dimension_mean - overall_dimension_mean)**2)
        denominator = numpy.sum(dimension_list.shape[1]*per_word_dimension_var)
        fisher_array[d] = numerator / denominator

    return fisher_array

def mutual_information(averaged_data, all_dimensions):

    ### Mutual Information
    mi_array = numpy.zeros(all_dimensions, dtype=numpy.double)
    for d in tqdm(range(all_dimensions)):
        dimension_list = numpy.array([[vec[d] for vec in v] for k, v in averaged_data.items()]).T
        combs = itertools.combinations(list(range(dimension_list.shape[0])), 2)
        dimension_mis = list()
        for c_i, c in enumerate(combs):
            var_one = numpy.var(dimension_list[c[0]])
            var_two = numpy.var(dimension_list[c[1]])
            cov = numpy.cov(numpy.array((dimension_list[c[0]], dimension_list[c[1]])))

            mi = math.log((var_one*var_two)/numpy.linalg.det(cov))/2
            dimension_mis.append(mi)
        dim_avg = numpy.nanmean(dimension_mis)
        mi_array[d] = dim_avg

    return mi_array

def stability(averaged_data, all_dimensions):

    ### Stability
    #dim_stability = list()
    stability_array = numpy.zeros(all_dimensions, dtype=numpy.double)
    for d in tqdm(range(all_dimensions)):
        dimension_list = numpy.array([[vec[d] for vec in v] for k, v in averaged_data.items()]).T
        combs = itertools.combinations(list(range(dimension_list.shape[0])), 2)
        dimension_corrs = list()
        for c_i, c in enumerate(combs):
            corr = stats.pearsonr(dimension_list[c[0]], dimension_list[c[1]])[0]
            dimension_corrs.append(corr)
        dim_avg = numpy.nanmean(dimension_corrs)
        #dim_stability.append((d, dim_avg))
        stability_array[d] = dim_avg
    #sorted_stability = sorted(dim_stability, key=lambda item : item[1], reverse=True)

    #return sorted_stability
    return stability_array

def attribute_correlation(averaged_data, all_dimensions, args, experiment):

    ### Attribute correlation
    # We start by averaging all ERPs
    averaged_data = {k : numpy.average(v, axis=0) for k, v in averaged_data.items()}
    # Loadin the model
    comp_model = WordVectors(args, experiment)
    comp_vectors = comp_model.vectors
    vector_dimensionality = list(set([v.shape for k, v in comp_vectors.items()]))[0][0]
    averaged_data = {experiment.trigger_to_info[k][0] : v for k, v in averaged_data.items()}
    comp_vectors = {k : comp_vectors[k] for k in averaged_data.keys()}
    #dim_stability = list()
    attribute_correlation_array = numpy.zeros(all_dimensions, dtype=numpy.double)
    for d in tqdm(range(all_dimensions)):
        erp_feature = [v[d] for k, v in averaged_data.items()]
        for v in range(vector_dimensionality):
            vector_feature = [vec[v] for k, vec in comp_vectors.items()]
            corr = stats.spearmanr(vector_feature, erp_feature)[0]
            attribute_correlation_array[d] += corr

    return attribute_correlation_array
