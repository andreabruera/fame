import os
import numpy
import collections

from tqdm import tqdm


def load_eeg_vectors(s, words, cats=[], categories=False):
    print('Now loading EEG Stanford vectors...')
    #mapping_dict = {'Dance': 'dancer', 'dancer' : 'Dance', 'pepper' : 'Capsicum', 'cow' : 'Cattle', 'grapes' : 'Grape', 'chili' : 'Chili_pepper', 'aubergine' : 'Eggplant', 'courgette' : 'Zucchini', 'baby bottle' : 'Baby_bottle', 'bulb' : 'Electric_light', 'cassette' : 'Cassette_tape', 'key' : 'Lock_and_key', 'phone' : 'Telephone'}     
    mapping_dict = {'Physical object' : 'Object'}
    final_vectors = collections.defaultdict(dict)
    #sub_folder = os.path.join('/import/cogsci/andrea/github/fame/data/neuro_data/eeg_stanford', 'sub_{:02}'.format(s))
    sub_folder = os.path.join('/import/cogsci/andrea/dataset/concatenated_vectors', 'sub_{:02}'.format(s))
    #sub_folder = os.path.join('/import/cogsci/andrea/github/fame/neuro_data/eeg_', 'sub_01'.format(s))
    sub_vectors = collections.defaultdict(list)
    for w in tqdm(words):
        if w in mapping_dict.keys():
            new_w = mapping_dict[w]
        with open(os.path.join(sub_folder, '{}.vec'.format(w))) as input_vector_file:
            vecs = [numpy.asarray(l.strip().split('\t'), dtype=numpy.single) for l in input_vector_file.readlines()]
        assert len(vecs) == 72
        for v in vecs:
            if categories:
                out_key = cats[w]
            else:
                out_key = w
            sub_vectors[out_key].append(v)
        '''
        counter = 0
        current_vector = []
        for v in vecs:
            if counter < 8:
                counter += 1
            else:
                counter = 0
                sub_vectors[w].append(numpy.average(current_vector, axis=0))
                current_vector = []
            current_vector.append(v)
        '''
    return sub_vectors        
