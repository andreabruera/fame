import collections
import numpy
import os
import scipy

from scipy import stats
from tqdm import tqdm


def write_plot_searchlight(args, n, explicit_times, results_array, hz=''):

    from io_utils import prepare_folder
    output_folder = prepare_folder(args)

    with open(os.path.join(output_folder, \
              '{}_{}sub-{:02}.rsa'.format(args.word_vectors, hz, n+1)), 'w') as o:
        for t in explicit_times:
            o.write('{}\t'.format(t))
        o.write('\n')
        for e in results_array:
            for t in e:
                o.write('{}\t'.format(t))
            o.write('\n')

    '''
    ### Plotting per-subject maps
    fig, ax = pyplot.subplots()
    mat = ax.imshow(results_array)
    
    ax.set_xticks([i+.5 for i in range(len(explicit_times))])
    ax.set_xticklabels(explicit_times, fontsize='xx-small')
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    pyplot.colorbar(mat, ax=ax)
    pyplot.savefig(os.path.join(output_folder, \
                   '{}_{}sub-{:02}_map.png'.format(args.word_vectors, hz, n+1)), \
                    dpi=1200)
    pyplot.clf()
    pyplot.close()
    '''

def join_searchlight_results(results, relevant_times):

    results_array = list()
    results_dict = {r[0] : r[1] for r in results}

    for e in range(128):
        e_row = list()
        for t in relevant_times:
            e_row.append(results_dict[(e, t)])
        results_array.append(e_row)

    results_array = numpy.array(results_array)

    return results_array

class SearchlightClusters:

    def __init__(self, max_distance=20):

        self.max_distance = max_distance
        self.index_to_code = self.indices_to_codes()
        self.neighbors = self.read_searchlight_clusters()
        self.mne_adjacency_matrix = self.create_adjacency_matrix()

    def create_adjacency_matrix(self):
        data = list()
        indices = list()
        index_pointer = [0]
        for i, kv in enumerate(self.neighbors.items()):
            v = kv[1][1:]
            for neighbor in v:
                indices.append(int(neighbor))
                data.append(1)
            index_pointer.append(len(indices))

        ### Just checking everything went fine
        mne_sparse_adj_matrix = scipy.sparse.csr_matrix((data, indices, index_pointer), dtype=int)
        for ikv, kv in enumerate(self.neighbors.items()):
            v = kv[1][1:]

            assert [i for i, k in enumerate(mne_sparse_adj_matrix.toarray()[ikv]) if k == 1] == v

        return mne_sparse_adj_matrix 

    def indices_to_codes(self):

        index_to_code = collections.defaultdict(str)
        with open('searchlight/searchlight_clusters_{}mm.txt'.format(self.max_distance), 'r') as searchlight_file:
            for l in searchlight_file:
                if 'CE' not in l:
                    l = l.strip().split('\t')
                    index_to_code[int(l[1])] = l[0]

        return index_to_code

    def read_searchlight_clusters(self):

        searchlight_clusters = collections.defaultdict(list)

        with open('searchlight/searchlight_clusters_{}mm.txt'.format(self.max_distance), 'r') as searchlight_file:
            for l in searchlight_file:
                if 'CE' not in l:
                    l = [int(i) for i in l.strip().split('\t')[1:]]
                    searchlight_clusters[l[0]] = l

        return searchlight_clusters


def run_searchlight(all_args): 

    eeg = all_args[0]
    comp_model = all_args[1] 
    cluster = all_args[2]
    word_combs = all_args[3]
    pairwise_similarities = all_args[4]
    step = all_args[5]

    places = list(cluster[0])
    start_time = cluster[1]

    eeg_similarities = list()

    for word_one, word_two in word_combs:

        eeg_one = eeg[word_one][places, start_time:start_time+(step*2)].flatten()
        eeg_two = eeg[word_two][places, start_time:start_time+(step*2)].flatten()
     
        ### Spearman Rho
        #word_comb_score = stats.spearmanr(eeg_one, eeg_two)[0]
        ### Pearson R
        word_comb_score = stats.pearsonr(eeg_one, eeg_two)[0]
        ### Mahalanobis distance
        '''
        inv_covariance = numpy.cov(numpy.array([eeg_one, eeg_two]).T).T
        word_comb_score = scipy.spatial.distance.mahalanobis(eeg_one, eeg_two, inv_covariance)
        '''
        ### Euclidean distance
        #word_comb_score = scipy.spatial.distance.euclidean(eeg_one, eeg_two)
        ### Cosine distance
        #word_comb_score = 1-scipy.spatial.distance.cosine(eeg_one, eeg_two)
        eeg_similarities.append(word_comb_score)

    ### Spearman rho
    #rho_score = scipy.stats.spearmanr(eeg_similarities, pairwise_similarities)[0]
    ### Pearson r
    rho_score = scipy.stats.pearsonr(eeg_similarities, pairwise_similarities)[0]
    #print('done with {} {}'.format(places[0], start_time))


    return [(places[0], start_time), rho_score]
