import matplotlib
import numpy
import os
import pdb

from matplotlib import font_manager, pyplot

### Creating the output folder
plot_folder = os.path.join('plots', 'one', 'clustering')
os.makedirs(plot_folder, exist_ok=True)

### Models

models = list()
labels = list()
purities = list()
vscores = list()
adj_rand = list()

for root, direc, filez in os.walk(os.path.join('results', 'one', 'clustering', 'computational_models')):
    for f in filez:
        if 'it' not in f:
            models.append(f)
            with open(os.path.join(root, f)) as i:
                lines = [l.split('\t') for l in i.readlines()]
            if len(labels) == 0:
                labels = [l[0]\
                          #.replace('_indiv', '\nindiv')
                           for l in lines]
                purities = [list() for l in lines]
                vscores = [list() for l in lines]
                adj_rand = [list() for l in lines]
            for l_i, l in enumerate(lines):
                purities[l_i].append(l[2])
                vscores[l_i].append(l[4])
                adj_rand[l_i].append(l[6])

models = [m.replace('_kmeans_cluster.scores', '').\
            replace('_en_mentions', '').\
            replace('original', '').\
            replace('_', ' ').\
            replace('tr', 'Tr').\
            replace('w', 'W').\
            replace('W2v', 'Word2Vec') for m in models \
            if 'vs' not in m\
            ]
 
### Reordering and numpying

#reordered_labels = sorted([l for l in enumerate(labels)], key=lambda item : item[1])
coarse = [l for l in labels if 'coarse' in l]
fine_places = [l for l in labels if 'places' in l]
fine_people = [l for l in labels if 'people' in l]
#control = [l for l in labels if 'vs' in l]
fine_all = [l for l in labels if l not in coarse+fine_places+fine_people\
           and 'vs' not in l #+control
           ]

reordered_labels = coarse + fine_all + fine_places + fine_people
                   # + control
indices = [labels.index(l) for l in reordered_labels]

labels = [labels[ind] for ind in indices]
purities = [purities[ind] for ind in indices]
vscores = [vscores[ind] for ind in indices]
adj_rand = [adj_rand[ind] for ind in indices]

purities = numpy.array(purities, dtype=numpy.single)
vscores = numpy.array(vscores, dtype=numpy.single)
adj_rand = numpy.array(adj_rand, dtype=numpy.single)

### Plotting the bars

width = .36

positions = [[i+(0.38*p)-.18 for i in range(1, 45, 5)] for p in range(\
             8
             #7\
             )]
below_positions = [i for i in range(1, 45, 5)]

### Font setup
# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

### Font size setup
SMALL_SIZE = 23
MEDIUM_SIZE = 25
BIGGER_SIZE = 27

pyplot.rc('font', size=SMALL_SIZE)          # controls default text sizes
pyplot.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
pyplot.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
pyplot.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pyplot.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pyplot.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

### Preparing a double plot
fig, ax = pyplot.subplots(nrows=2, ncols=1, \
                          gridspec_kw={'height_ratios': [6, 1]}, \
                          figsize=(16,9),\
                          constrained_layout=True)

ax[0].set_ylim(bottom=-.1, top=1.1)
ax[0].tick_params(axis='x', which='both', bottom=False, top=False, \
                  labelbottom=False, labeltop=False)
ax[1].set_ylim(bottom=0, top=.5)
ax[0].hlines(y=0., xmin=0., xmax=50., alpha=0.)
ax[1].hlines(y=0.4, xmin=0, xmax=50., alpha=0.)
ax[1].spines['top'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)

ax[0].get_xaxis().set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
ax[0].set_ylabel('Adjusted Rand Index', fontweight='bold', labelpad=10.)

for m_i, m in enumerate(models):
    xs = positions[m_i]
    ax[0].bar(xs, adj_rand[:, m_i], label=m, width=width)

ax[0].legend(ncol=4, fontsize='x-small', bbox_to_anchor=(0.86, 1.15),\
             frameon=False)
#ax[0].set_xticks(range(1, 20, 2))
#ax[0].set_xticklabels(labels)
#pyplot.setp(ax[0].get_xticklabels(), rotation=45, fontsize=6, ha="right", rotation_mode="anchor")


ax[1].text(s='Coarse-grained', x=7, y=.5, \
                     ha='center', va='center', \
                     fontweight='bold')
ax[1].text(s='Fine-grained', x=30, y=.5,\
                      ha='center', va='center', \
                     fontweight='bold')
#ax[1].text(s='Control', x=48.5, y=.5, \
#              ha='center', va='center',\
#                      fontweight='bold')
for x, y in [(0.1, 14), (16, \
            44)
            #50.9), 
             #(46, 50.9)
            ]:

    ax[1].hlines(y=.4, xmin=x, \
                 xmax=y, color='gray', \
                 linestyle='dashdot', alpha=0.7, \
                 )

for p, l in zip(below_positions, labels):
    text = l.replace('_only', '').\
             replace('categories', 'C').\
             replace('individuals', 'I').\
             replace('and', '+').\
             replace('people_', 'people\n').\
             replace('places', 'places\n').\
             replace('fine_', '').\
             replace('coarse_', '').\
             replace('_', ' ') if 'vs' not in l else 'I vs C'
    ax[1].text(s=text, x=p+1.75, y=.2, ha='center', va='center', \
               #fontweight='bold'
               )

ax[1].text(s='I = categories', x=7.75, y=-.1, ha='left', va='center', fontweight='bold')
ax[1].text(s='C = categories', x=25.75, y=-.1, ha='left', va='center', fontweight='bold')

#pyplot.savefig(os.path.join(plot_folder, 'computational_models_cluster_bars.pdf'), dpi=800)
pyplot.savefig(os.path.join(plot_folder, 'computational_models_cluster_bars.jpg'), dpi=600)
import pdb; pdb.set_trace()
### Plotting the violins

fig, ax = pyplot.subplots(constrained_layout=True)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylabel('Adjusted Rand Index')
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels)
pyplot.setp(ax.get_xticklabels(), rotation=45, fontsize=6, ha="right", rotation_mode="anchor")
ax.violinplot(adj_rand.T, showmedians=True, showextrema=False)
#pyplot.show()
pyplot.savefig(os.path.join(plot_folder, 'computational_models_cluster_violins.png'), dpi=600)


### Subjects

labels = list()
purities = list()
vscores = list()
adj_rand = list()

for root, direc, filez in os.walk(os.path.join('clustering', 'erp')):

    for f in filez:
        with open(os.path.join(root, f)) as i:
            lines = [l.split('\t') for l in i.readlines()]
        if len(labels) == 0:
            labels = [l[0].replace('_ind', '\nind') for l in lines]
            purities = [list() for l in lines]
            vscores = [list() for l in lines]
            adj_rand = [list() for l in lines]
        for l_i, l in enumerate(lines):
            purities[l_i].append(l[2])
            vscores[l_i].append(l[4])
            adj_rand[l_i].append(l[6])

### Reordering and numpying
labels = [labels[ind] for ind, v in reordered_labels]
purities = [purities[ind] for ind, v in reordered_labels]
vscores = [vscores[ind] for ind, v in reordered_labels]
adj_rand = [adj_rand[ind] for ind, v in reordered_labels]

purities = numpy.array(purities, dtype=numpy.single)
vscores = numpy.array(vscores, dtype=numpy.single)
adj_rand = numpy.array(adj_rand, dtype=numpy.single)

### Plotting
fig, ax = pyplot.subplots(constrained_layout=True)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylabel('Adjusted Rand Index')
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels)
pyplot.setp(ax.get_xticklabels(), rotation=45, fontsize=6, ha="right", rotation_mode="anchor")
ax.violinplot(adj_rand.T, showmedians=True, showextrema=False)
pyplot.savefig(os.path.join(plot_folder, 'erp_cluster_violins.png'), dpi=600)
#pyplot.show()

