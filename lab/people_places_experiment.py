import psychopy
import numpy
import os
import re
import time
import pickle 

from utils import read_words, read_trigger_ids, select_words, prepare_runs
from messages import ExperimentOne

from psychopy import gui, visual, event, parallel

experiment = 'one'
exp_text = ExperimentOne()
###############
##### Getting ready
###############

# Setting the parallel port for EEG triggers

port_number = 888
outputPort = parallel.ParallelPort(port_number)
outputPort.setData(0)

### subject, actual runs to be ran (out of 32), refresh rate of the screen
gui = gui.Dlg()
gui.addField("Subject ID:")
gui.show()

subjectId = int(gui.data[0])

### make window
### TO DO: adapting screen size to whatever size so as to avoid crashes/bugs
#win = visual.Window(size=[1920,1080], fullscr=False, color=[-1,-1,-1], units='pix', winType='pyglet', screen=0, checkTiming=True)
win = psychopy.visual.Window(size=[1920,1080], fullscr=True, color=[-1,-1,-1], units='pix', checkTiming=True)
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
subjectPath = os.path.join('results', timeNow, 'sub-{:02}_events'.format(subjectId))
os.makedirs(subjectPath, exist_ok=True) 
print(subjectPath)

### Loading the experiment data

words_and_cats = read_words(experiment=experiment)

selected_words_and_cats = select_words(words_and_cats)
        
runs, questions, answers, triggers = prepare_runs(selected_words_and_cats, experiment=experiment)
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

for r in range(24):
#for r in range(22, 24):
    outputPort.setData(0)

    ### Preparing the file
    with open(os.path.join(subjectPath, 'run_{:02}.events'.format(r)), 'w') as o:
        o.write('Word\tTrigger ID\tQuestion type\tCorrect answer\tAnswer given\tResponse time\n')

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
        outputPort.setData(0)
        clock = psychopy.core.Clock()
        while clock.getTime()< 1.:
            textStimulus.text = '+'
            win.flip()
    
        ### Target word
        outputPort.setData(int(trigger)) # Sending the EEG trigger, opening the parallel port with the trigger ID
        #print(trigger)
        #print(word)
        clock = psychopy.core.Clock()
        while clock.getTime()< .75:
            textStimulus.text = word.replace('Ã' ,'à') ### Ad-hoc correction
            win.flip()
        #stimulusDuration = clock.getTime() # stores stimulus presentation duration
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

        if responseKey == answer:
            accuracy += 1

        question_type = 0 if 'riferiva' not in question else 1

        run_results.append([word, trigger, question_type, answer, responseKey, responseTime])

    overall_results.append(run_results)

    ### Writing to file run details
    with open(os.path.join(subjectPath, 'run_{:02}.events'.format(r)), 'a') as o:
        for t in run_results:
            for info in t:
                if info != t[-1]:
                    o.write('{}\t'.format(info))
                else:
                    o.write('{}\n'.format(info))
    
    ### Inter-run break message

    if r < 23:
        accuracy = accuracy/.4
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
        while clock.getTime()< 3.:
            textStimulus.text = 'Overall accuracy: {}%'.format(numpy.average(competition_list))
            win.flip()

### Pickling all results
with open(os.path.join(subjectPath, 'all_results.pkl'), 'wb') as o:
    pickle.dump(overall_results, o)
