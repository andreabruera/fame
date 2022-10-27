import nipype
import os

from nipype.interfaces import spm

spm_path = os.path.join('..', 'resources', 'spm12_standalone', \
                        'spm12', 'run_spm12.sh')
matlab_compiler_path = os.path.join('..', 'resources', 'spm12_standalone', \
                        'MATLAB_Compiler_Runtime', 'v713')
matlab_cmd = '{} {} script'.format(spm_path, matlab_compiler_path)
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

import pdb; pdb.set_trace()
