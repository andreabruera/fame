import psychopy
import numpy
import os
import re
import time
import pickle 

from utils import read_words_and_triggers, prepare_runs
from messages import ExperimentOne

from psychopy import gui, visual, event

experiment = 'two'
exp_text = ExperimentOne()

###############
##### Getting ready
###############

### subject, actual runs to be ran (out of 32), refresh rate of the screen
gui_one = gui.Dlg()
gui_one.addField("Subject ID:")
gui_one.addField('In the lab? [y/n]')
gui_one.show()

s = int(gui_one.data[0])
in_the_lab = gui_one.data[1]

### Settin the parallel port
if in_the_lab not in ['y', 'n']:
    raise RuntimeError('Wrong answer regarding parallel port')

if in_the_lab == 'y':
    in_the_lab = True
    # Setting the parallel port for EEG triggers
    from psychopy import parallel
    port_number = 888
    outputPort = parallel.ParallelPort(port_number)
    outputPort.setData(0)
else:
    in_the_lab = False

familiar_words_and_triggers = dict()

### Familiar people

for i in range(1, 9):

    gui_two = gui.Dlg()
    gui_two.addField("Nome della persona {}: ".format(i))
    gui_two.addField('Categoria: ')
    gui_two.show()
    
    familiar_words_and_triggers[gui_two.data[0]] = ['persona', gui_two.data[1], i]
    
### Familiar places

for i in range(11, 19):

    gui_three = gui.Dlg()
    gui_three.addField("Nome del luogo {}: ".format(i-10))
    gui_three.addField('Categoria: ')
    gui_three.show()
    
    familiar_words_and_triggers[gui_three.data[0]] = ['luogo', gui_three.data[1], i]

### make window
### TO DO: adapting screen size to whatever size so as to avoid crashes/bugs

win = psychopy.visual.Window(size=[1920,1080], fullscr=True, color=[-1,-1,-1], units='pix', checkTiming=True)
#win = psychopy.visual.Window(size=[720, 480], fullscr=False, color=[-1,-1,-1], units='pix', checkTiming=True)
loadingMessage = 'L\'esperimento si sta caricando...'
textStimulus = visual.TextStim(win, loadingMessage, color=[.8,.8,.8], pos=[0,0], ori=0, wrapWidth=1080, height=40, contrast=.5, font='Calibri')

### Printing the loading message

textStimulus.autoDraw = True

### Printing out to output the screen setting, for debugging

print('Frame rate detected: {}'.format(win.getActualFrameRate()))
actualFrameMs = win.getMsPerFrame(nFrames=180)[2]
predictedFrameMs = win.monitorFramePeriod
print('Predicted milliseconds per frame: {}'.format(predictedFrameMs*1000.))
print('Actual milliseconds per frame: {}'.format(actualFrameMs))

### Setting and creating the output folder

timeNow = time.strftime('%d_%b_%Hh%M', time.gmtime())
subjectPath = os.path.join('results', timeNow, 'sub-{:02}_events'.format(s))
os.makedirs(subjectPath, exist_ok=True) 
print(subjectPath)
### Loading the experiment data

famous_words_and_triggers = read_words_and_triggers()

### Reuniting the dictionaries
final_words_and_triggers = dict()

for w, lst in familiar_words_and_triggers.items():
    final_words_and_triggers[w] = lst
for w, lst in famous_words_and_triggers.items():
    final_words_and_triggers[w] = lst

assert len(final_words_and_triggers.items()) == 32

runs, questions, answers, triggers = prepare_runs(final_words_and_triggers)

competition_list = list()

###############
### Instructions
###############

textStimulus.text = exp_text.presentation

win.flip()
event.waitKeys(keyList=['space'])

textStimulus.text = exp_text.organization
win.flip()
event.waitKeys(keyList=['space'])

textStimulus.text = exp_text.recommendations
win.flip()
event.waitKeys(keyList=['space'])

textStimulus.text = exp_text.start
win.flip()
event.waitKeys(keyList=['space'])

##############
#### Experiment
##############

overall_results = list()

for r in range(32):
    if in_the_lab:
        outputPort.setData(0)

    run = runs[r]
    question_list = questions[r]
    answer_list = answers[r]
    trigger_list = triggers[r]

    run_results = list()
    accuracy = 0

    for t in range(len(run)):

        word = run[t]
        question = question_list[t]
        answer = answer_list[t]
        trigger = trigger_list[t]

        ### Fixation
        if in_the_lab:
            outputPort.setData(0)
        clock = psychopy.core.Clock()
        while clock.getTime()< 1.2:
            textStimulus.text = '+'
            win.flip()
    
        ### Target word
        if in_the_lab:
            outputPort.setData(int(trigger)) # Sending the EEG trigger, opening the parallel port with the trigger ID
            #print(trigger)
        #print(word)
        clock = psychopy.core.Clock()
        while clock.getTime()< .5:
            textStimulus.text = word.replace('Ã' ,'à') ### Ad-hoc correction
            win.flip()
        #stimulusDuration = clock.getTime() # stores stimulus presentation duration
        if in_the_lab:
            outputPort.setData(0) # Closing the parallel port

        ### Fixation again
        clock = psychopy.core.Clock()
        while clock.getTime()< 1.:
            textStimulus.text = '+'
            win.flip()

        ### Question
        textStimulus.text = question.replace('Ã' ,'à') ### Ad-hoc correction
        win.flip()
        clock = psychopy.core.Clock()
        response_data = event.waitKeys(keyList=['s','k'], timeStamped=clock)
        responseKey, responseTime = response_data[0][0], response_data[0][1]
        print([answer,  responseKey])
        if responseKey == answer:
            accuracy += 1
            print('correct')

        question_type = 0 if 'riferiva' not in question else 1

        run_results.append([word, trigger, question_type, answer, responseKey, responseTime])

    overall_results.append(run_results)

    ### Preparing the file

    with open(os.path.join(subjectPath, 'sub-{:02}_run-{:02}.events'.format(s, r+1)), 'w') as o:
        o.write('Word\tTrigger ID\tQuestion type\tCorrect answer\tAnswer given\tResponse time\n')

        ### Writing to file run details
        for t in run_results:
            for info in t:
                if info != t[-1]:
                    o.write('{}\t'.format(info))
                else:
                    o.write('{}\n'.format(info))
    
    ### Inter-run break message

    if r < 23:
        accuracy = accuracy/.32
        competition_list.append(accuracy)
        for countdown in range(30, -1, -1):
            clock = psychopy.core.Clock()
            while clock.getTime()< 1.:
                textStimulus.text = exp_text.break_message([r, accuracy, countdown])
                win.flip()

        textStimulus.text = exp_text.start
        win.flip()
        event.waitKeys(keyList=['space'])

    ### End
    else:
        clock = psychopy.core.Clock()
        while clock.getTime()< 3.:
            textStimulus.text = exp_text.end
            win.flip()
        clock = psychopy.core.Clock()
        while clock.getTime()< 30.:
            textStimulus.text = 'Overall accuracy: {}%'.format(numpy.average(competition_list))
            win.flip()

        textStimulus.text = exp_text.start
        win.flip()
        k = event.waitKeys(keyList=['space', 'q'])
        if k[0] == 'q':
            break

### Pickling all results
with open(os.path.join(subjectPath, 'all_results.pkl'), 'wb') as o:
    pickle.dump(overall_results, o)