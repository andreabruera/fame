from utils import read_words, select_words

experiment = 'one'

words_and_cats = read_words(experiment=experiment)

selected_words_and_cats = select_words(words_and_cats)
print(selected_words_and_cats)
'''
c = 101
with open('exp_two_stimuli_triggers.txt', 'w', encoding='utf-8') as o: 
    o.write('Word\tCoarse category\tFine category\tTrigger\n')
    for coarse, coarse_dict in selected_words_and_cats.items():
        for fine, fine_list in coarse_dict.items():
            fine = fine.replace('citt', 'città')
            for f in fine_list:
                o.write('{}\t{}\t{}\t{}\n'.format(f, coarse, fine, c))
                c += 1
'''