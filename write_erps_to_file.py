import os

from io_utils import ExperimentInfo, LoadEEG

class Args:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.analysis = 'decoding'
        self.corrected = False
        self.data_kind = 'erp'
        self.subsample = 'subsample_10'
        self.average = 24
        self.data_folder = os.path.join(
                                        '/',
                                        'import',
                                        'cogsci',
                                        'andrea',
                                        'dataset',
                                        'neuroscience',
                                        )
        if self.experiment_id == 'two':
            self.entities = 'individuals_and_categories'
            self.data_folder = os.path.join(
                                            self.data_folder,
                                            'family_lexicon_eeg',
                                            #'derivatives'
                                            )
            self.semantic_category = 'famous'
        else:
            self.entities = 'individuals_only'
            self.data_folder = os.path.join(
                                            self.data_folder,
                                            'exploring_individual_entities_eeg',
                                            #'derivatives'
                                            )
            self.semantic_category = 'all'


for experiment_id in ['one', 'two']:

    args = Args(experiment_id)

    with open('rough_and_ready_brain_data_exp_{}.eeg'.format(experiment_id), 'w') as o:
        o.write('subject\tstimulus\teeg_vector\n')
        for sub in range(1, 34):
            experiment = ExperimentInfo(
                                        args, 
                                        subject=sub
                                        )
            eeg_data = LoadEEG(args, experiment, sub)
            eeg = {experiment.trigger_to_info[k][0] : v.flatten() for k, v in eeg_data.data_dict.items()}
            for k, v in eeg.items():
                o.write('{}\t{}\t'.format(sub, k))
                for dim in v:
                    o.write('{}\t'.format(dim))
                o.write('\n')
