from nni.experiment import Experiment
from pathlib import Path
import os,sys
import time
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

params = {'batch_size': 32,'hidden_size1': 128,'hidden_size2': 128, 'lr': 0.001,'momentum': 0.5}

search_space = {
    'batch_size': {'_type': 'choice', '_value': [32, 64, 128, 256]},
    'hidden_size1': {'_type': 'choice', '_value': [64, 128, 256, 512]},
    'hidden_size2': {'_type': 'choice', '_value': [64, 128, 256, 512]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}
experiment = Experiment('local')

experiment.config.trial_code_directory = '.'
experiment.config.trial_command = 'python mnist_nni.py'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 30
experiment.config.trial_gpu_number = 0
experiment.config.trial_concurrency = 2
experiment.run(8080)

input('Press enter to quit')
experiment.stop()