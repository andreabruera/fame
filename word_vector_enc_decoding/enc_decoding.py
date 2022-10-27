import os
import argparse
import numpy

from io_utils import ExperimentInfo, prepare_folder
from read_word_vectors import WordVectors
from encoding_decoding_utils import prepare_and_test

if args.word_vectors == 'bert':
    args.extraction_method = 'full_sentence'
elif args.word_vectors == 'elmo':
    args.extraction_method = 'unmasked'
elif args.word_vectors == 'ernie':
    args.extraction_method = 'full_sentence'

if __name__ == '__main__':
    
    '''
    if args.analysis == 'decoding_only_cats':
        words_to_fine = {v[0] : v[2] for k, v in experiment.trigger_to_info.items()}
        original_vecs = comp_vectors.copy()
        comp_vectors = {k : original_vecs[words_to_fine[k]] for k, v in original_vecs.items()}
    '''


    out_path = prepare_folder(args)
    time_marker = 'time_resolved' if args.time_resolved else 'whole_epoch'
    random_marker = 'random' if args.random else 'true_evaluation'
    out_path = os.path.join(out_path, args.restrict_words, \
                            time_marker, random_marker)
    os.makedirs(out_path, exist_ok=True)

    ### Recording the word vector layer if needed
    if args.word_vectors == 'bert' or args.word_vectors == 'ernie':
        file_name = '{}_{}_layer_{}_{}_results.txt'.format(\
                     args.word_vectors, \
                     args.extraction_method, args.layer, \
                     args.evaluation_method)
    elif args.word_vectors == 'elmo':
        file_name = '{}_{}_{}_results.txt'.format(args.word_vectors, \
                     args.extraction_method, \
                     args.evaluation_method)
    else:
        file_name = '{}_{}_results.txt'.format(args.word_vectors, \
                     args.evaluation_method)


    if not args.time_resolved:
        ### Writing to file
        with open(os.path.join(out_path, file_name), 'w') as o:
            o.write('Accuracy\t')
            for w in word_by_word.keys():
                o.write('{}\t'.format(w))
            o.write('\n')
            for a_i, a in enumerate(accuracies):
                o.write('{}\t'.format(a))
                for w, accs in word_by_word.items():
                    acc = accs[a_i]
                    o.write('{}\t'.format(acc))
                o.write('\n')

    else:
        ### Writing to file timepoint by timepoint
        with open(os.path.join(out_path, file_name), 'w') as o:
            for t in times:
                o.write('{}\t'.format(t))
            o.write('\n')
            for a in accuracies:
                for t_acc in a:
                    o.write('{}\t'.format(t_acc))
                o.write('\n')

        if args.word_vectors == 'bert' or args.word_vectors == 'ernie':
            word_file_name = 'word_by_word_{}_{}_layer_{}_{}.txt'.format(\
                              args.word_vectors, \
                              args.extraction_method, \
                              args.layer, args.evaluation_method)
        elif args.word_vectors == 'elmo':
            word_file_name = 'word_by_word_{}_{}_{}.txt'.format(\
                              args.word_vectors, \
                              args.extraction_method, \
                              args.evaluation_method)
        else:
            word_file_name = 'word_by_word_{}_{}.txt'.format(\
                              args.word_vectors, \
                              args.evaluation_method)
        with open(os.path.join(out_path, word_file_name), 'w') as o:
            for w in word_by_word.keys():
                o.write('{}\t'.format(w))
            o.write('\n')
            for i in range(n_subjects):
                for w, values in word_by_word.items():
                    o.write('{}\t'.format(values[i]))
                o.write('\n')
