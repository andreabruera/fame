import gensim
import itertools
import math
import numpy
import os
import pickle
import re
import scipy
import sklearn

from gensim.models import KeyedVectors, Word2Vec
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec

def cluster_scores(input_samples, labels):

    labels = sklearn.preprocessing.LabelEncoder().fit_transform(y=[cat \
                                     for w, cat in labels])
    clusterer = KMeans(len(list(set(labels))))
    #clusterer = DBSCAN(len(list(set(labels))))
    #clusterer = SpectralClustering(len(list(set(labels))))
    preds = clusterer.fit_predict(input_samples)

    purity_score = purity(preds, labels)
    v_score = sklearn.metrics.v_measure_score(labels, preds)
    adjusted_rand_score = sklearn.metrics.adjusted_rand_score(labels, preds)

    scores = (round(purity_score, 4), round(v_score, 4), round(adjusted_rand_score, 4))

    return scores

def purity(preds, labels):

    counter = {l : list() for l in list(set(labels))}
    for p, l in zip(preds, labels):
        counter[l].append(p)
    mapper = {m_k : {p : len([v for v in m_v if v == p]) for p in list(set(m_v))} for m_k, m_v in counter.items()}
    mapper = {k : max(v.items(), key=lambda item : item[1])[0] for k, v in mapper.items()}

    scores = 0
    for p, l in zip(preds, labels):
        if p == mapper[l]:
            scores += 1
    purity_score = scores / len(labels)

    return purity_score  

def compute_clusters(vectors, trigger_to_info):

    cluster_results = dict()

    ### Individuals vs categories
    coarse_labels = [(v[0], 0) if k<=100 else (v[0], 1) for k, v in trigger_to_info.items()]
    input_samples = [vectors[w] for w, coarse in coarse_labels]

    cluster_results['individuals_vs_categories'] = cluster_scores(input_samples, coarse_labels)

    ### All together

    ### Coarse-grained
    coarse_labels = [(v[0], v[1]) for k, v in trigger_to_info.items()]
    input_samples = [vectors[w] for w, coarse in coarse_labels]

    cluster_results['coarse_individuals_and_categories'] = cluster_scores(input_samples, coarse_labels)

    ### Fine-grained
    fine_labels = [(v[0], v[2]) for k, v in trigger_to_info.items()]
    input_samples = [vectors[w] for w, coarse in fine_labels]

    cluster_results['fine_individuals_and_categories'] = cluster_scores(input_samples, fine_labels)

    ### Fine-grained people only
    fine_labels = [(v[0], v[2]) for k, v in trigger_to_info.items() if v[1] == 'persona']
    input_samples = [vectors[w] for w, coarse in fine_labels]

    cluster_results['fine_people_individuals_and_categories'] = cluster_scores(input_samples, fine_labels)

    ### Fine-grained places only
    fine_labels = [(v[0], v[2]) for k, v in trigger_to_info.items() if v[1] == 'luogo']
    input_samples = [vectors[w] for w, coarse in fine_labels]

    cluster_results['fine_places_individuals_and_categories'] = cluster_scores(input_samples, fine_labels)

    ### Individuals only
    ### Coarse-grained
    coarse_labels = [(v[0], v[1]) for k, v in trigger_to_info.items() \
                                                           if k <= 100]
    input_samples = [vectors[w] for w, coarse in coarse_labels]

    cluster_results['coarse_individuals_only'] = cluster_scores(input_samples, coarse_labels)

    ### Fine-grained
    fine_labels = [(v[0], v[2]) for k, v in trigger_to_info.items() \
                                                          if k <= 100]
    input_samples = [vectors[w] for w, coarse in fine_labels]

    cluster_results['fine_individuals_only'] = cluster_scores(input_samples, fine_labels)
    ### Fine-grained people only
    fine_labels = [(v[0], v[2]) for k, v in trigger_to_info.items() if v[1] == 'persona' and k <= 100]
    input_samples = [vectors[w] for w, coarse in fine_labels]

    cluster_results['fine_people_individuals_only'] = cluster_scores(input_samples, fine_labels)

    ### Fine-grained places only
    fine_labels = [(v[0], v[2]) for k, v in trigger_to_info.items() if v[1] == 'luogo' and k <= 100]
    input_samples = [vectors[w] for w, coarse in fine_labels]

    cluster_results['fine_places_individuals_only'] = cluster_scores(input_samples, fine_labels)

    ### Categories only
    ### Coarse-grained
    coarse_labels = [(v[0], v[1]) for k, v in trigger_to_info.items() \
                                                           if k > 100]
    input_samples = [vectors[w] for w, coarse in coarse_labels]

    cluster_results['coarse_categories_only'] = cluster_scores(input_samples, coarse_labels)

    return cluster_results

class WordVectors:
 
    def __init__(self, args, experiment, s=''):
    
        self.experiment = experiment
        self.sub = s
        self.words = self.word_mapper(args)

        self.word_to_trigger = {v[0] : k for k, v in experiment.trigger_to_info.items()}
        '''
        if args.entities == 'categories_only':
            self.words = {k : v for k, v in self.words.items() if self.word_to_trigger[v] > 100}
            assert len(self.words.keys()) == 8
        elif args.entities == 'individuals_only':
            self.words = {k : v for k, v in self.words.items() if self.word_to_trigger[v] <= 100}
            assert len(self.words.keys()) == 32 
        else:
            assert len(self.words.keys()) == 40 

        ### Restricting to people or places only
        if args.semantic_category == 'people':
            self.words = {k : v for k, v in self.words.items() if experiment.trigger_to_info[self.word_to_trigger[v]][1] == 'persona'}
            assert len(self.words.keys()) in [16, 20]

        elif args.semantic_category == 'places':
            self.words = {k : v for k, v in self.words.items() if experiment.trigger_to_info[self.word_to_trigger[v]][1] == 'luogo'}
            assert len(self.words.keys()) in [16, 20]
        '''
        #self.words = list(self.word_to_trigger.keys())
        self.words = {k : v for k, v in self.words.items() if v in [v[0] for v in self.experiment.trigger_to_info.values()]}
        assert len(self.words.keys()) in [16, 20, 32, 40]

        '''
        ## Fix for subject 21, who had already visited South Africa
        if self.sub == 21:
            del self.words['Sud Africa']
            self.words['Canada'] = ['Canada']
        elif self.sub == 22:
            del self.words['Piramidi di Giza']
            self.words['Machu Picchu'] = ['Macchu Picchu']
        '''
            
        #print(self.words)
        self.vectors = self.read_vectors(args)
        #self.ordered_words, \
        #    self.word_combs, \
        #    self.pairwise_similarities = self.compute_pairwise()
        cluster_file_path = os.path.join('results', args.experiment_id, \
                                         'clustering', 'computational_models')
        os.makedirs(cluster_file_path, exist_ok=True)
        cluster_file_path  = os.path.join(cluster_file_path, \
                             '{}_kmeans_cluster.scores'.format(\
                                             args.word_vectors))
        self.ordered_words, self.combs, self.pairwise_similarities = self.compute_pairwise()

        if not os.path.exists(cluster_file_path):
            try:
                cluster_results = compute_clusters(self.vectors, self.experiment.trigger_to_info)
                with open(cluster_file_path, 'w') as o:
                    for k, v in cluster_results.items():
                        o.write('{}\tpurity\t{}\tv-score\t{}\tadjusted rand score\t{}\n'.format(k, v[0], v[1], v[2]))
            except (RuntimeError, KeyError):
                print('Could not run clustering analysis for the current model')
                pass

    def compute_pairwise(self):
        
        ordered_words = sorted([v for k, v in self.words.items()])
        #ordered_words = sorted(list(self.vectors.keys()))
        for w in ordered_words:
            assert w in self.vectors.keys()
        combs = list(itertools.combinations(ordered_words, 2))
        pairwise_similarities = list()
        for c in combs:
            ### Spearman rho
            sim = stats.spearmanr(self.vectors[c[0]], self.vectors[c[1]])[0]
            ### Inverse Spearman rho
            #sim = 1 - stats.spearmanr(self.vectors[c[0]], self.vectors[c[1]])[0]
            ### Pearson R
            #sim = stats.pearsonr(self.vectors[c[0]], self.vectors[c[1]])[0]
            ### Cosine distance
            #sim = scipy.spatial.distance.cosine(self.vectors[c[0]], self.vectors[c[1]])
            ### Cosine similarity
            #sim = 1. - scipy.spatial.distance.cosine(self.vectors[c[0]], self.vectors[c[1]])
            pairwise_similarities.append(sim)

        lengths = list()
        for c in combs:
            dist = abs(len(c[0]) - len(c[1]))
            lengths.append(dist)
        ### Correlation with lengths
        length_correlation = scipy.stats.spearmanr([1-s for s in pairwise_similarities], lengths)
        #length_correlation = scipy.stats.linregress([1-s for s in pairwise_similarities], lengths).rvalue

        print('correlation with length: {}'.format(length_correlation))

        ### Turning word combs to trigger combs
        combs = [tuple([self.word_to_trigger[c[i]] for i in range(2)]) for c in combs]

        #out_folder = os.path.join('results', 'one', 'models_pairwise')
        #os.makedirs(out_folder, exist_ok=True)
        #with open(os.path.join(out_folder, 
        
        return ordered_words, combs, pairwise_similarities

    def read_vectors(self, args):
    
        if args.word_vectors == 'gpt2':
            vectors = self.read_gpt2(args)
        if args.word_vectors == 'bert' or \
        args.word_vectors == 'ernie':
            vectors = self.read_bert(args)
        if args.word_vectors == 'elmo':
            vectors = self.read_elmo(args)
        if args.word_vectors == 'w2v':
            vectors = self.read_w2v(args)
        if args.word_vectors == 'it_w2v':
            vectors = self.read_it_w2v(args)
        if args.word_vectors == 'transe':
            vectors = self.read_transe(args)
        if args.word_vectors == 'wikipedia2vec':
            vectors = self.read_wikipedia2vec(args)
        if args.word_vectors == 'it_wikipedia2vec':
            vectors = self.read_it_wikipedia2vec(args)
        if 'BERT' in args.word_vectors or \
                'LUKE' in args.word_vectors or \
                'ELMO' in args.word_vectors or \
                'GPT' in args.word_vectors:
            vectors = self.read_BERT(args)

        '''
        ### Vector isotropy correction as in Mu & Viswanath 2018
        average = numpy.average([v for k, v in vectors.items()], axis=0)
        vectors = {k : v-average for k, v in vectors.items()}
        pca = sklearn.decomposition.PCA(n_components=0.99)
        pca.fit([v for k, v in vectors.items()])
        pca_component = numpy.zeros(numpy.array([v for k,v in vectors.items()]).shape)
        for i in range(4):
            zero = pca.components_[i]
            pca_component += (numpy.array([v for k, v in vectors.items()]) * zero.T)*zero
        vectors = {kv[0] : kv[1]-pca_vec for kv, pca_vec in zip(vectors.items(), pca_component)}
        #vectors = {kv[0] : pca_vec for kv, pca_vec in zip(vectors.items(), pca_component)}
        '''
        if args.wv_dim_reduction != 'no_dim_reduction':
            amount = int(args.wv_dim_reduction[-2:])/100
            pca = sklearn.decomposition.PCA(n_components=amount)
            vecs = pca.fit_transform([v for k, v in vectors.items()])
            vectors = {k : pca_vec for k, pca_vec in zip(vectors.keys(), vecs)}
    
        return vectors

    def read_gpt2(self, args):

        ### reading file
        with open('exp_{}_gpt2_vectors.tsv'.format(args.experiment_id)) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        vectors = {l[0] : numpy.array(l[1:], dtype=numpy.float64) for l in lines}
        for k, v in vectors.items():
            assert v.shape == (1024, )
        word_vectors = dict()
        print('Now reading ITGPT2 word vectors...')
        for new, original in tqdm(self.words.items()):
            word_vectors[new] = vectors[original]
        
        return word_vectors

    def read_BERT(self, args):

        word_vectors = dict()
        print('Now reading word vectors...')
        for new, original in tqdm(self.words.items()):
            if self.sub == '':
                file_name = os.path.join('word_vectors', args.word_vectors, \
                                         args.experiment_id, \
                                         '{}.vec'.format(original))
            else:
                file_name = os.path.join('word_vectors', args.word_vectors, \
                                         args.experiment_id, \
                                         'sub-{:02}'.format(self.sub), \
                                         '{}.vec'.format(new))
            with open(file_name) as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            if 'ELMO' in args.word_vectors:
                lines = lines[-1]
                current_vec = numpy.array(lines, dtype=numpy.double)
            elif 'BERT' in args.word_vectors or 'LUKE' in args.word_vectors:
                if 'large' in args.word_vectors:
                    #lines = lines[14:18]
                    lines = lines[-8:]
                else:
                    lines = lines[-4:]

                current_vec = numpy.array(lines, dtype=numpy.double)
                current_vec = numpy.average(current_vec, axis=0)

            if 'BERT' in args.word_vectors or 'LUKE' in args.word_vectors:
                if not 'large' in args.word_vectors:
                    assert current_vec.shape == (768, )
                else:
                    assert current_vec.shape == (1024, )
            elif 'ELMO' in args.word_vectors:
                if not 'original' in args.word_vectors:
                    assert current_vec.shape == (256, )
                else:
                    assert current_vec.shape == (1024, )
            
            word_vectors[original] = current_vec

        return word_vectors

    def read_it_wikipedia2vec(self, args):

        ### Loading pickle if possible
        os.makedirs(os.path.join('word_vectors', 'it_wikipedia2vec'), exist_ok=True)
        pickle_file_name = os.path.join('word_vectors', 'it_wikipedia2vec', \
                                        'it_wikipedia2vec_pickle.pkl')
        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as i:
                word_vectors = pickle.load(i)
        else:
            model_path = '/import/cogsci/andrea/dataset/word_vectors/wikipedia2vec/itwiki_20180420_300d.pkl'
            wikipedia2vec = Wikipedia2Vec.load(model_path)

            w2v_mapping = {'Sagrada_Família' : 'Sagrada Familia', \
                           'Madonna' : 'Madonna (cantante)', \
                           'J.K. Rowling' : 'J. K. Rowling', \
                           'Macchu Picchu' : 'Machu Picchu', \
                           'Marylin Monroe' : 'Marilyn Monroe', \
                           "corso d'acqua" : ['fiume', 'lago', \
                                              'mare', 'oceano']}
            missing = list()
            word_vectors = dict()
            for w, original in tqdm(self.words.items()):
                #vocabulary_name = '/en/{}'.format(w.lower().replace(' ', '_'))
                if original in w2v_mapping.keys():
                    original_alias = w2v_mapping[original]
                else:
                    original_alias = original
                if not isinstance(original_alias, list):
                    original_alias = [original_alias]

                alias_list = list()
                for query_original in original_alias:
                    
                    try:
                        w_vec = wikipedia2vec.get_entity_vector(query_original)
                        alias_list.append(w_vec)
                    except KeyError:
                        try:
                            w_vec = wikipedia2vec.get_word_vector(query_original)
                            alias_list.append(w_vec)
                        except KeyError:
                            missing.append(query_original)
                        
                if len(alias_list) > 1:
                    w_vec = numpy.average(alias_list, axis=0)
                elif len(alias_list) == 1:
                    w_vec = alias_list[0]
                else:
                    print(original)
                assert w_vec.shape == (300, )
                
                word_vectors[original] = w_vec
            try:
                assert len(missing) == 0
            except AssertionError:
                import pdb; pdb.set_trace()

            ### Pickling if file not there
            with open(pickle_file_name, 'wb') as o:
                pickle.dump(word_vectors, o)

        return word_vectors

    def read_wikipedia2vec(self, args):

        ### Loading pickle if possible
        os.makedirs(os.path.join('word_vectors', 'wikipedia2vec'), exist_ok=True)
        pickle_file_name = os.path.join('word_vectors', 'wikipedia2vec', \
                                        'wikipedia2vec_pickle.pkl')
        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as i:
                word_vectors = pickle.load(i)
        else:
            model_path = '/import/cogsci/andrea/dataset/word_vectors/wikipedia2vec/enwiki_20180420_win10_500d.pkl'
            wikipedia2vec = Wikipedia2Vec.load(model_path)

            w2v_mapping = {'Sagrada_Família' : 'Sagrada Familia'}
            missing = list()
            word_vectors = dict()
            for w, original in tqdm(self.words.items()):
                if w in w2v_mapping.keys():
                    w = w2v_mapping[w]
                #vocabulary_name = '/en/{}'.format(w.lower().replace(' ', '_'))
                
                try:
                    w_vec = wikipedia2vec.get_entity_vector(w)
                except KeyError:
                    missing.append(w)
                word_vectors[original] = w_vec
            assert len(missing) == 0

            ### Pickling if file not there
            with open(pickle_file_name, 'wb') as o:
                pickle.dump(word_vectors, o)

        return word_vectors

    def read_it_w2v(self, args):

        ### Loading pickle if possible
        os.makedirs(os.path.join('word_vectors', 'it_w2v'), exist_ok=True)
        pickle_file_name = os.path.join('word_vectors', 'it_w2v', \
                                        'it_w2v_pickle.pkl')
        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as i:
                word_vectors = pickle.load(i)
        else:
            w2v_model = Word2Vec.load('/import/cogsci/andrea/dataset/word_vectors/w2v_it_wexea/w2v_it_wexea')

            w2v_mapping = {'Sagrada Familia': 'Sagrada Família', \
                           'Madonna' : 'Madonna cantante', \
                           'J.K. Rowling' : 'J K Rowling', \
                           'Macchu Picchu' : 'Machu Picchu', \
                           'Marylin Monroe' : 'Marilyn Monroe', \
                           "corso d'acqua" : ['fiume', 'lago', \
                                              'mare', 'oceano']}
            missing = list()
            word_vectors = dict()
            for w, original in tqdm(self.words.items()):
                #vocabulary_name = '/en/{}'.format(w.lower().replace(' ', '_'))
                if original in w2v_mapping.keys():
                    original_alias = w2v_mapping[original]
                else:
                    original_alias = original
                if not isinstance(original_alias, list):
                    original_alias = [original_alias]

                alias_list = list()
                for query_original in original_alias:
                    
                    query_original = query_original.lower().replace(' ', '_')
                    try:
                        w_vec = w2v_model[query_original]
                        alias_list.append(w_vec)
                    except KeyError:
                        missing.append(query_original)
                        
                if len(alias_list) > 1:
                    w_vec = numpy.average(alias_list, axis=0)
                elif len(alias_list) == 1:
                    w_vec = alias_list[0]
                else:
                    print(original)
                assert w_vec.shape == (300, )
                
                word_vectors[original] = w_vec
            try:
                assert len(missing) == 0
            except AssertionError:
                import pdb; pdb.set_trace()

            ### Pickling if file not there
            with open(pickle_file_name, 'wb') as o:
                pickle.dump(word_vectors, o)

        return word_vectors

    def read_w2v(self, args):

        ### Loading pickle if possible
        os.makedirs(os.path.join('word_vectors', 'w2v'), exist_ok=True)
        pickle_file_name = os.path.join('word_vectors', 'w2v', \
                                        'w2v_pickle.pkl')
        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as i:
                word_vectors = pickle.load(i)
        else:
            model_path = '/import/cogsci/andrea/dataset/word_vectors/freebase-vectors-skipgram1000-en.bin'
            w2v = KeyedVectors.load_word2vec_format(model_path, binary=True)

            w2v_mapping = {'J. K. Rowling': 'J K Rowling', \
                           'Hillary Clinton' : 'Hillary Rodham Clinton', \
                           'Freddie Mercury' : 'Queen', \
                           'Madonna (entertainer)' : 'Madonna', \
                           'Sagrada_Família' : 'Sagrada Familia'}
            missing = list()
            word_vectors = dict()
            for w, original in tqdm(self.words.items()):
                if w in w2v_mapping.keys():
                    w = w2v_mapping[w]
                vocabulary_name = '/en/{}'.format(w.lower().replace(' ', '_'))
                try:
                    w_vec = w2v[vocabulary_name]
                except KeyError:
                    missing.append(w)
                word_vectors[original] = w_vec
            assert len(missing) == 0

            ### Pickling if file not there
            with open(pickle_file_name, 'wb') as o:
                pickle.dump(word_vectors, o)

        return word_vectors

    def read_transe(self, args):

        ### Loading pickle if possible
        pickle_file_name = os.path.join('word_vectors', 'transe', \
                                               'transe_pickle.pkl')
        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as i:
                word_vectors = pickle.load(i)
        else:
    
            word_vectors = dict()

            print('Now reading word vectors...')
            for new, original in tqdm(self.words.items()):
                current_word_vectors = list()

                file_name = os.path.join('word_vectors', 'transe', \
                                         '{}.vec'.format(new.replace(' ', '_')))
                with open(file_name) as i:
                    lines = [l.strip().split('\t') for l in i.readlines()][1]
                
                current_vec = numpy.array(lines, dtype=numpy.double)
                assert current_vec.shape == (100, )
                
                word_vectors[original] = current_vec

            ### Pickling if file not there
            with open(pickle_file_name, 'wb') as o:
                pickle.dump(word_vectors, o)

        return word_vectors

    def read_elmo(self, args):

        ### Loading pickle if possible
        pickle_file_name = os.path.join('word_vectors', 'elmo', \
                                        'elmo_pickle_unmasked.pkl')
        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as i:
                word_vectors = pickle.load(i)
        else:
    
            word_vectors = dict()

            print('Now reading word vectors...')
            for new, original in tqdm(self.words.items()):
                current_word_vectors = list()

                file_name = os.path.join('word_vectors', 'elmo', 'unmasked', \
                                         '{}.vec'.format(new.replace(' ', '_')))
                with open(file_name) as i:
                    lines = [l.strip().split('\t') for l in i.readlines()]
                indices = list(range(len(lines)))[1::2]
                
                for index in indices:
                    current_vec = numpy.array(lines[index], dtype=numpy.double)
                    current_word_vectors.append(current_vec)
                current_word_vectors = numpy.average(numpy.array(current_word_vectors[:24]), axis=0)
                assert current_word_vectors.shape == (1024, )
                
                word_vectors[original] = current_word_vectors

            ### Pickling if file not there
            with open(pickle_file_name, 'wb') as o:
                pickle.dump(word_vectors, o)

        return word_vectors

    def read_bert(self, args):

        ### Loading pickle if possible
        pickle_file_name = os.path.join('word_vectors', args.word_vectors, \
                                        'layer_{}_pickle_unmasked.pkl'.format(\
                                        args.layer))
        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as i:
                word_vectors = pickle.load(i)
        else:
    
            word_vectors = dict()

            print('Now reading word vectors...')
            for new, original in tqdm(self.words.items()):
                current_word_vectors = list()

                file_name = os.path.join('word_vectors', args.word_vectors, \
                                         args.extraction_method, \
                                         '{}.vec'.format(new.replace(' ', '_')))
                with open(file_name) as i:
                    lines = [l.strip().split('\t') for l in i.readlines()]
                indices = list(range(len(lines)))[1::13]
                
                for index in indices:
                    current_vec = numpy.array(lines[index:index+12], dtype=numpy.double)
                    current_word_vectors.append(current_vec)
                current_word_vectors = numpy.average(numpy.array(current_word_vectors[:6]), axis=0)
                assert current_word_vectors.shape == (12, 768)
                current_word_vectors = numpy.average(current_word_vectors, axis=0)
                assert current_word_vectors.shape == (768, )
                
                word_vectors[original] = current_word_vectors

            ### Pickling if file not there
            with open(pickle_file_name, 'wb') as o:
                pickle.dump(word_vectors, o)

        return word_vectors

    def word_mapper(self, args):

        if args.experiment_id == 'two':
            with open('lab/lab_two/stimuli/exp_two_stimuli_triggers.txt') as i:
                words = [l.strip().split('\t')[0] for l in i.readlines()][1:]
            words = words + list(range(1, 9)) + list(range(11, 19))
            eeg_mapper = {k : k for k in words}

        if args.experiment_id == 'one':
            with open('lab/stimuli/final_words_exp_one.txt') as ids_txt:
                lines = [l.strip().split('\t')[0] for l in ids_txt.readlines()]

            names_mapper = {'Papa Francesco' : 'Pope Francis', 'Gandhi' : 'Mahatma Gandhi', \
                            'Martin Luther King' : 'Martin Luther King Jr.', \
                            'J.K. Rowling' : 'J. K. Rowling', 'Lev Tolstoj' : 'Leo Tolstoy', \
                            'Marylin Monroe' : 'Marilyn Monroe', 'Germania' : 'Germany', \
                            'Israele' : 'Israel', 'Giappone' : 'Japan', 'Svizzera' : 'Switzerland', \
                            'Regno Unito' : 'United Kingdom', 'Spagna' : 'Spain', \
                            'Sud Africa' : 'South Africa', 'Sud Corea' : 'South Korea', \
                            'Pechino' : 'Beijing', 'Rio De Janeiro' : 'Rio de Janeiro', \
                            'Atene' : 'Athens', 'Oceano Pacifico' : 'Pacific Ocean', \
                            'Canale della Manica' : 'English Channel', \
                            'Mar Mediterraneo' : 'Mediterranean Sea', 'Mar dei Caraibi' : 'Caribbean Sea', \
                            'Mar Nero' : 'Black Sea', 'Nilo' : 'Nile', 'Parigi' : 'Paris',\
                            'Mare del Nord' : 'North Sea', 'Baia di Hudson' : 'Hudson Bay', \
                            'Oceano Atlantico' : 'Atlantic Ocean', 'Roma' : 'Rome',\
                            'Monte Rushmore' : 'Mount Rushmore', 'Piramidi di Giza' : 'Giza pyramid complex', \
                            'Muraglia Cinese' : 'Great Wall of China', 'Mosca' : 'Moscow',\
                            'Macchu Picchu' : 'Machu Picchu', 'Monte Saint-Michel' : 'Mont-Saint-Michel', \
                            'Torre di Pisa' : 'Leaning Tower of Pisa', 'Mare Rosso' : 'Red Sea',\
                            'Sagrada Familia' : 'Sagrada_Família', 'New York' : 'New York City',\
                            'Madonna' : 'Madonna (entertainer)', \
                            'musicista' : 'Musician', 'attore' : 'Actor', \
                            'politico' : 'Politician', 'scrittore' : 'Writer', \
                            'corso d\'acqua' : 'Body of water', 'città' : 'City', \
                            'monumento' : 'Monument', 'stato' : 'Country'}

            eeg_mapper = dict()
            for l in lines:
                
                if l in names_mapper.keys():
                    name = names_mapper[l]
                else:
                    name = l

                eeg_mapper[name] = l

        return eeg_mapper

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def load_vectors_two(args, experiment, n, clustered=False):

    names = [v[0] for v in experiment.trigger_to_info.values()]
    if args.word_vectors == 'ceiling':
        ceiling = dict()
        for n_ceiling in range(1, 34):
            eeg_data_ceiling = LoadEEG(args, experiment, n_ceiling)
            eeg_ceiling = eeg_data_ceiling.data_dict
            for k, v in eeg_ceiling.items():
                ### Adding and flattening
                if k not in ceiling.keys():
                    ceiling[k] = list()
                ceiling[k].append(v.flatten())
        comp_vectors = {k : numpy.average([vec for vec_i, vec in enumerate(v) if vec_i!=n-1], axis=0) for k, v in ceiling.items()}
        comp_vectors = {experiment.trigger_to_info[k][0] : v for k, v in comp_vectors.items()}

    elif args.word_vectors in ['coarse_category', 'famous_familiar', 'fine_category']:
        if args.word_vectors == 'coarse_category':
            categories = {v[0] : v[1] for v in experiment.trigger_to_info.values()}
        elif args.word_vectors == 'famous_familiar':
            if args.experiment_id == 'one':
                raise RuntimeError('There is no famous_familiar distinction for this experiment!')
            categories = {v[0] : v[2] for v in experiment.trigger_to_info.values()}
        elif args.word_vectors == 'fine_category':
            if args.experiment_id == 'two':
                raise RuntimeError('There is no famous_familiar distinction for this experiment!')
            categories = {v[0] : v[2] for v in experiment.trigger_to_info.values()}
        #vectors = {k_one : numpy.array([0. if categories[k_one]==categories[k_two] else 1. for k_two in names if k_one != k_two]) for k_one in names}
        sorted_categories = sorted(set(categories.values()))
        vectors = {w : sorted_categories.index(categories[w]) for w in names}

    elif args.word_vectors == 'word_length':
        #lengths = [len(w) for w in names]
        #vectors = {k_one : numpy.array([abs(l_one-l_two) for k_two, l_two in zip(names, lengths) if k_two!=k_one]) for k_one, l_one in zip(names, lengths)}
        vectors = {w : len(w) for w in names}
        vectors = zero_one_norm(vectors)

    elif args.word_vectors == 'orthography':
        ### vector of differences
        #vectors = {k_one : numpy.array([levenshtein(k_one,k_two) for k_two in names if k_two!=k_one]) for k_one in names}
        ### average of differences
        vectors = {k_one : numpy.average([levenshtein(k_one,k_two) for k_two in names if k_two!=k_one]) for k_one in names}
        vectors = zero_one_norm(vectors)

    elif args.word_vectors in ['imageability', 'familiarity']:

        fams = list()
        if args.experiment_id == 'two' and args.semantic_category == 'familiar':
            ### considering length of documents as familiarity index
            fams_dict = dict()
            sent_folder = os.path.join('personally_familiar_sentences')
            sub_marker = 'sub-{:02}'.format(n)
            for f in os.listdir(sent_folder):
                if sub_marker in f:
                    f_path = os.path.join(sent_folder, f)
                    with open(f_path) as i:
                        words = [w for l in i.readlines() for w in l.split()]
                    fam_name = re.sub(r'{}_|\.sentences'.format(sub_marker), '', f)
                    fams_dict[fam_name.replace('_', ' ').replace('89 1', '89/1').strip()] = len(words)
            for name in names:
                assert name in fams_dict.keys()
            fams = [fams_dict[n] for n in names]
        else:
            filename = os.path.join('lab','stimuli',
                                    '{}_ratings_experiment.csv'.format(args.word_vectors))
            with open(filename) as i:
                lines = [l.strip().split('\t')[1:] for l in i.readlines()]
            assert len(names) <= len(lines[0])
            for k in names:
                try:
                    assert k in lines[0]
                    rel_index = lines[0].index(k)
                    try:
                        fam = numpy.average([int(l[rel_index]) for l in lines[1:]])
                    except IndexError:
                        fam = float(lines[-1][-1])
                except AssertionError:
                    fam = 3.5
                fams.append(fam)

        #vectors = {k_one : numpy.array([abs(l_one-l_two) for k_two, l_two in zip(names, fams) if k_two!=k_one]) for k_one, l_one in zip(names, fams)}
        vectors = {k_one : l_one for k_one, l_one in zip(names, fams)}
        vectors = zero_one_norm(vectors)

    elif 'frequency' in args.word_vectors:
        freqs = list()
        for k in names:
            with open(os.path.join('entity_sentences_{}_from_all_corpora'.format(args.experiment_id), 'it', '{}.sentences'.format(k))) as i:
                lines = [l.strip() for l in i.readlines()]
            lines = [l for l in lines if len(l) > 3]
            freqs.append(len(lines))
        if args.word_vectors == 'log_frequency':
            #vectors = {k_one : numpy.array([abs(math.log(l_one)-math.log(l_two)) for k_two, l_two in zip(names, freqs) if k_two!=k_one]) for k_one, l_one in zip(names, freqs)}
            vectors = {k_one : math.log(l_one) for k_one, l_one in zip(names, freqs)}
        else:
            vectors = {k_one : float(l_one) for k_one, l_one in zip(names, freqs)}
            #vectors = {k_one : numpy.array([abs(l_one-l_two) for k_two, l_two in zip(names, freqs) if k_two!=k_one]) for k_one, l_one in zip(names, freqs)}
        vectors = zero_one_norm(vectors)

    elif args.word_vectors == 'w2v':

        w2v_model = Word2Vec.load('/import/cogsci/andrea/dataset/word_vectors/w2v_it_wexea/w2v_it_wexea')

        w2v_mapping = {'Sagrada Familia': 'Sagrada Família', \
                       'Madonna' : 'Madonna cantante', \
                       'J.K. Rowling' : 'J K Rowling', \
                       'Macchu Picchu' : 'Machu Picchu', \
                       'Marylin Monroe' : 'Marilyn Monroe', \
                       "corso d'acqua" : ['fiume', 'lago', \
                                          'mare', 'oceano']}
        vectors = dict()
        for original in names:
            #vocabulary_name = '/en/{}'.format(w.lower().replace(' ', '_'))
            if original in w2v_mapping.keys():
                original_alias = w2v_mapping[original]
            else:
                original_alias = original
            if not isinstance(original_alias, list):
                original_alias = [original_alias]

            alias_list = list()
            for query_original in original_alias:
                
                query_original = query_original.lower().replace(' ', '_')
                try:
                    w_vec = w2v_model[query_original]
                except KeyError:
                    individual_words = query_original.split('_')
                    w_vec = numpy.average([w2v_model[w] for w in individual_words], axis=0)

                alias_list.append(w_vec)
                    
            if len(alias_list) > 1:
                w_vec = numpy.average(alias_list, axis=0)
            elif len(alias_list) == 1:
                w_vec = alias_list[0]
            else:
                print(original)
            assert w_vec.shape == (300, )
            
            vectors[original] = w_vec

    elif args.word_vectors in [ 
                               'gpt2', 
                               'xlm-roberta-large',
                               'MBERT',
                             ]:
        ### reading file
        #with open('exp_{}_{}_wikipedia_vectors.tsv'.format(args.experiment_id, args.word_vectors)) as i:
        with open('exp_{}_{}_vectors.tsv'.format(args.experiment_id, args.word_vectors)) as i:
            lines = [l.strip().split('\t') for l in i.readlines()][1:]
        vectors = {l[0] : numpy.array(l[1:], dtype=numpy.float64) for l in lines}
        for k, v in vectors.items():
            #print(v.shape)
            assert v.shape in [(1024, ), (768,)]
    else:
        vectors = dict()
        folder = os.path.join('word_vectors_10_2022', 
                              args.experiment_id,
                              args.word_vectors,
                              'top_four', 
                              #'top_six', 
                              #'first_four', 
                              #'middle_four',
                              'span_averaged',
                              )
        assert os.path.exists(folder)
        files = [f for f in os.listdir(folder)]
        assert len(files) in [16, 18, 32, 40]
        for f in tqdm(files):

            file_path = os.path.join(folder, f)
            with open(file_path) as i:
                lines = [l.strip().split('\t') for l in i]
                try:
                    #assert len(lines) in [1, 20]
                    assert len(lines) >= 1 and len(lines) <= 50
                except AssertionError:
                    print('error with {}'.format(f))
                    continue
                if clustered:
                    vecs = numpy.array(lines, dtype=numpy.float64)
                else:
                    vecs = numpy.array(lines[0], dtype=numpy.float64)
            entity = re.sub('sub-\d\d\_|\.vector', '', f)
            vectors[entity.replace('_', ' ')] = vecs

    return vectors

def zero_one_norm(vectors):
    values = [v for k, v in vectors.items()]
    values = [(v - min(values))/(max(values)-min(values)) for v in values]
    vectors = {k[0] : val for k, val in zip(vectors.items(), values)}
    return vectors
