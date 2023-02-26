# -*- coding: latin-1 -*-

import os
import re
import collections
import numpy
import random

def read_words(experiment, additional_folder=''):

    familiarity_path = os.path.join(additional_folder, 'stimuli', 'familiarity_ratings_experiment_{}.csv'.format(experiment))
    with open(familiarity_path) as i:
        fam_lines = [l.strip().split('\t')[1:] for l in i.readlines()]
    #print(fam_lines)

    familiarity = dict()
    for i in range(len(fam_lines[0])):
        entity = fam_lines[0][i]
        fam = fam_lines[-1][i]
        #print(entity)
        familiarity[entity] = float(fam.replace(',', '.'))

    list_path = os.path.join(additional_folder, 'stimuli', 'exp_{}_stimuli.txt'.format(experiment))
    with open(list_path) as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]
    words_and_cats = {l[1] : (l[2], l[3], familiarity[l[1]]) for l in lines}

    return words_and_cats

def prepare_mentions(experiment, additional_folder=''):

    list_path = os.path.join(additional_folder, 'stimuli', 'exp_{}_stimuli.txt'.format(experiment))
    with open(list_path) as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]
    mentions = {l[1] : l[4] for l in lines}

    return mentions

def read_trigger_ids(experiment, additional_folder=''):

    list_path = os.path.join('stimuli', 'exp_{}_stimuli.txt'.format(experiment))
    with open(list_path) as i:
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
    selected_fine = {k : v[:(int(len(v)/2)-1)] for k, v in fine_cats.items()}

    coarse_cats = {v[1] : v[0] for k, v in words_and_cats.items()}
    averages = collections.defaultdict(list)
    for k, v in selected_fine.items():
        coarse = coarse_cats[k]
        average = (k,  numpy.average([fam[1] for fam in v]))
        averages[coarse].append(average)

    average = {k : sorted(v, key=lambda item:item[1], reverse=True) for k, v in averages.items()}
    fine_cats_to_remove = [v[-1][0] for k, v in average.items()]

    for k in fine_cats_to_remove:
        selected_fine.pop(k)

    final_dict = collections.defaultdict(lambda : collections.defaultdict(list))
    for k, v in selected_fine.items():
        coarse = coarse_cats[k]
        fine = k
        entities = [fam[0] for fam in v]
        final_dict[coarse][fine] = entities

    return final_dict

def prepare_runs(selected_words, experiment):

    entities = [e for c, c_dict in selected_words.items() for f, e_list in c_dict.items() for e in e_list]
    fine = [f for c, c_dict in selected_words.items() for f, e_list in c_dict.items()]

    ent_to_fine = {e : f for c, c_dict in selected_words.items() for f, e_list in c_dict.items() for e in e_list}
    fine_to_coarse = {f : c for c, c_dict in selected_words.items() for f in c_dict.keys()}

    
    all_stimuli = entities + fine
    ready_stimuli = random.sample(all_stimuli, k=len(all_stimuli))

    runs = list()
    for i in range(24):
        runs.append(random.sample(ready_stimuli, k=len(ready_stimuli)))

    questions = list()
    answers = list()
    triggers = list()

    trigger_ids = read_trigger_ids(experiment)
    mentions = prepare_mentions(experiment)
    balanced_questions = [0 for i in range(8)] + [1 for i in range(24)]
    balanced_correct_wrong = [0 for i in range(16)] + [1 for i in range(16)]
    question_one = 'Persona [S]\t\t\to\t\t\tLuogo [K]?'
    question_two = lambda word : 'La parola si riferiva ad {}.\n\nCorretto [S]\t\t\t\tSbagliato [K]'.format(mentions[word])

    for run in runs:

        run_questions = list()
        run_answers = list()
        run_triggers = list()

        shuffled_questions = random.sample(balanced_questions, k=len(balanced_questions))
        shuffled_correct_wrong = random.sample(balanced_correct_wrong, k=len(balanced_correct_wrong))
        c_q = 0
        c_q_w = 0

        for i, w in enumerate(run):

            trigger = trigger_ids[w]

            if w in fine:
                question = question_one
                answer = 's' if fine_to_coarse[w]=='persona' else 'k'
            else:
                question = shuffled_questions[c_q]
                c_or_w = shuffled_correct_wrong[c_q_w]

                c_q += 1
                c_q_w += 1

                if question == 0:
                    question = question_one
                    answer = 's' if fine_to_coarse[ent_to_fine[w]]=='persona' else 'k'
                else:
                    corresponding_fine = ent_to_fine[w]
                    if c_or_w == 1:
                        e = w
                        answer = 's'
                    else:
                        e = random.choice([e for e, f in ent_to_fine.items() if f!=corresponding_fine])
                        answer = 'k'
                    question = question_two(e)

            run_questions.append(question)
            run_answers.append(answer)
            run_triggers.append(trigger)

        questions.append(run_questions)
        answers.append(run_answers)
        triggers.append(run_triggers)

    for i, r in enumerate(runs):
        for i_w, w in enumerate(r):
            assert triggers[i][i_w] == trigger_ids[w]

    return runs, questions, answers, triggers
