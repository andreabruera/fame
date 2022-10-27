import matplotlib
import numpy
import os
import pdb
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats

### Font setup
# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

### Font size setup
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

pyplot.rc('font', size=SMALL_SIZE)          # controls default text sizes
pyplot.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
pyplot.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
pyplot.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pyplot.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pyplot.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

folder = 'plots/one/decoding_results_breakdown/all_time_points/individuals_and_categories/people_and_places/'
out_folder = os.path.join('plots', 'one', 'vector_analyses')
os.makedirs(out_folder, exist_ok=True)

files = [f for f in os.listdir(folder) if 'txt' in f]

results = dict()

for f in files:
    with open(os.path.join(folder, f)) as i:
        lines = [l.strip().split('\t') for l in i.readlines()]
    model = f.replace('_individuals_and_categories_people_and_places_decoding_breakdown.txt', '')
    results[model] = numpy.array(lines[1:], dtype=numpy.double)

sorted_keys = sorted(list(results.keys()))
sorted_results = {k.replace('_', ' ') : results[k] for k in sorted_keys}
sorted_keys = sorted_results.keys()

### Computing general score correlation

correlations = list()
for k, v in sorted_results.items():
    relevant_slice_one = v[:, 0]
    row = list()
    for k_two, v_two in sorted_results.items():
        relevant_slice_two = v_two[:, 0]
        correlation = stats.spearmanr(relevant_slice_one, relevant_slice_two)[0]
        row.append(correlation)
    correlations.append(row)

title = 'Pairwise correlations between models when considering overall accuracy'

fig, ax = pyplot.subplots(figsize=(16, 9), constrained_layout=True)
ax.imshow(correlations)
for i in range(len(sorted_keys)):
    for j in range(len(sorted_keys)):
        ax.text(i, j, round(correlations[i][j], 2), ha='center', va='center')

ax.set_xticks(range(len(sorted_keys)))
ax.set_xticklabels(sorted_keys, rotation=45, ha='right')
ax.set_yticks(range(len(sorted_keys)))
ax.set_yticklabels(sorted_keys)
ax.set_title(title, fontweight='bold', pad=20.)

pyplot.savefig(os.path.join(out_folder, \
               'overall_accuracy_correlations.pdf'), dpi=600)
pyplot.clf()

### Computing breakdown scores correlation

correlations = list()
for k, v in sorted_results.items():
    relevant_slice_one = numpy.average(v, axis=0)
    row = list()
    for k_two, v_two in sorted_results.items():
        relevant_slice_two = numpy.average(v_two, axis=0)
        correlation = stats.spearmanr(relevant_slice_one, relevant_slice_two)[0]
        row.append(correlation)
    correlations.append(row)

title = 'Pairwise correlations between models when considering breakdown of results'

fig, ax = pyplot.subplots(figsize=(16, 9), constrained_layout=True)
ax.imshow(correlations)
for i in range(len(sorted_keys)):
    for j in range(len(sorted_keys)):
        ax.text(i, j, round(correlations[i][j], 2), ha='center', va='center')

ax.set_xticks(range(len(sorted_keys)))
ax.set_xticklabels(sorted_keys, rotation=45, ha='right')
ax.set_yticks(range(len(sorted_keys)))
ax.set_yticklabels(sorted_keys)
ax.set_title(title, fontweight='bold', pad=20.)

pyplot.savefig(os.path.join(out_folder, \
               'results_breakdown_correlations.pdf'), dpi=600)
pyplot.clf()
