import os
import re
import collections
import numpy
import random

def read_words_and_triggers():
    with open(os.path.join('stimuli', 'exp_two_stimuli_triggers.txt'), errors='ignore') as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]
    words_and_triggers = {l[0] : [l[1], l[2], int(l[3])] for l in lines}
    
    return words_and_triggers

def read_words(experiment):

    familiarity_path = os.path.join('stimuli', 'familiarity_ratings_experiment_{}.csv'.format(experiment))
    with open(familiarity_path, errors='ignore') as i:
        fam_lines = [l.strip().split('\t')[1:] for l in i.readlines()]
    #print(fam_lines)

    print(fam_lines)
    familiarity = dict()
    for i in range(len(fam_lines[0])):
        entity = fam_lines[0][i]
        fam = fam_lines[-1][i]
        #print(entity)
        familiarity[entity] = float(fam.replace(',', '.'))

    list_path = os.path.join('stimuli', 'exp_{}_stimuli.txt'.format(experiment))
    with open(list_path, errors='ignore') as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]
    words_and_cats = {l[1] : (l[2], l[3], familiarity[l[1]]) for l in lines}
    ### Edit for experiment 2
    excluded_words = ['Torre di Pisa', 'Parigi', 'Roma', 'Atene', 'Woody Allen', 'Quentin Tarantino', 'Istanbul', \
                                     'Silvio Berlusconi', 'Papa Francesco', 'Svizzera', 'Cervino', 'Mar Mediterraneo', 'Nilo', \
                                      'Sagrada Familia', 'Amsterdam', 'Spagna', 'Germania', 'Regno Unito', 'Marylin Monroe', \
                                      'Bob Dylan', 'Angelina Jolie', "corso d'acqua", 'atleta', 'Mosca', 'Sydney']
    words_and_cats = {k : v for k, v in words_and_cats.items() if k not in excluded_words and v[1] not in excluded_words}

    return words_and_cats

def prepare_mentions(experiment):

    list_path = os.path.join('stimuli', 'exp_{}_stimuli.txt'.format(experiment))
    with open(list_path, errors='ignore') as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]
    mentions = {l[1] : l[4] for l in lines}

    return mentions

def read_trigger_ids(experiment):

    list_path = os.path.join('stimuli', 'exp_{}_stimuli.txt'.format(experiment))
    with open(list_path, errors='ignore') as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]
    trigger_ids = {l[1] : l[0] for l in lines}
    fine_cats = sorted(list(set([l[3] for l in lines])))
    for i in range(len(fine_cats)):
        fine_cat = fine_cats[i]
        i += 101
        trigger_ids[fine_cat] = i

    return trigger_ids

def select_words(words_and_cats):
    
    fine_cats = collections.defaultdict(list)
    for word, cats in words_and_cats.items():
        fine_cat = cats[1]
        fam = cats[2]
        fine_cats[fine_cat].append((word, fam))
    fine_cats = {k : sorted(v, key=lambda item : item[1], reverse=True) for k, v in fine_cats.items()}
    #selected_fine = {k : v[:(int(len(v)/2)-1)] for k, v in fine_cats.items()}
    selected_fine = {k : v[:2] for k, v in fine_cats.items()}

    coarse_cats = {v[1] : v[0] for k, v in words_and_cats.items()}
    averages = collections.defaultdict(list)
    for k, v in selected_fine.items():
        coarse = coarse_cats[k]
        average = (k,  numpy.average([fam[1] for fam in v]))
        averages[coarse].append(average)

    average = {k : sorted(v, key=lambda item:item[1], reverse=True) for k, v in averages.items()}
    '''
    fine_cats_to_remove = [v[-1][0] for k, v in average.items()]

    for k in fine_cats_to_remove:
        selected_fine.pop(k)
    '''

    final_dict = collections.defaultdict(lambda : collections.defaultdict(list))
    for k, v in selected_fine.items():
        coarse = coarse_cats[k]
        fine = k
        entities = [fam[0] for fam in v]
        final_dict[coarse][fine] = entities

    return final_dict

def prepare_runs(stimuli_dict):

    entities = list(stimuli_dict.keys())
    coarse_lst = list(set([e_lst[0] for e, e_lst in stimuli_dict.items()]))
    fine_lst = list(set([e_lst[1] for e, e_lst in stimuli_dict.items()]))

    ready_stimuli = random.sample(entities, k=len(entities))

    runs = list()
    for i in range(32):
        runs.append(random.sample(ready_stimuli, k=len(ready_stimuli)))

    questions = list()
    answers = list()
    triggers = list()

    balanced_questions = [0 for i in range(8)] + [1 for i in range(24)]
    balanced_correct_wrong = [0 for i in range(16)] + [1 for i in range(16)]
    question_one = 'Persona [S]\t\t\to\t\t\tLuogo [K]?'
    question_one_inverted = 'Luogo [S]\t\t\to\t\t\tPersona [K]?'
    question_two = lambda word : 'La parola si riferiva ad {}.\n\nCorretto [S]\t\t\t\tSbagliato [K]'.format(word)

    for run in runs:

        run_questions = list()
        run_answers = list()
        run_triggers = list()

        shuffled_questions = random.sample(balanced_questions, k=len(balanced_questions))
        shuffled_correct_wrong = random.sample(balanced_correct_wrong, k=len(balanced_correct_wrong))
        c_q = 0
        c_q_w = 0

        for i, w in enumerate(run):
            coarse = stimuli_dict[w][0]
            fine = stimuli_dict[w][1]
            trigger = stimuli_dict[w][2]
            '''
            if w in fine:
                chosen_index = random.choice([0, 1])
                if chosen_index == 0:
                    question = question_one
                    answer = 's' if fine_to_coarse[w]=='persona' else 'k'
                else:
                    question = question_one_inverted
                    answer = 'k' if fine_to_coarse[w]=='persona' else 's'
            else:
            '''
            question = shuffled_questions[c_q]
            c_or_w = shuffled_correct_wrong[c_q_w]

            c_q += 1
            c_q_w += 1

            if question == 0: # coarse question
                chosen_index = random.choice([0, 1])
                if chosen_index == 0:
                    question = question_one
                    answer = 's' if coarse=='persona' else 'k'
                else:
                    question = question_one_inverted
                    answer = 'k' if coarse=='persona' else 's'
            else: # fine question
                if c_or_w == 1:
                    f = fine
                    answer = 's'
                else:
                    f = random.choice([f for f in fine_lst if f!=fine and f[:3]!=fine[:3]])
                    answer = 'k'
                question = question_two(f)

            run_questions.append(question)
            run_answers.append(answer)
            run_triggers.append(trigger)

        questions.append(run_questions)
        answers.append(run_answers)
        triggers.append(run_triggers)

    for i, r in enumerate(runs):
        for i_w, w in enumerate(r):
            assert triggers[i][i_w] == stimuli_dict[w][-1]

    return runs, questions, answers, triggers
