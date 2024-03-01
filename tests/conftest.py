import datetime
from src.data_generation import sklearn_make_moons
import pytest
import os

# environment vars for preventing long runtime and repeats
NUM_REPEATS      = 10
MAX_TIMEOUT_SECS = 60 * 60

# environment variables defining repeatability and input size/noise/num_clusters
RAND_SEED       = 42
# INPUT_SIZES     = [x for x in range(100, 3001, 100)]
INPUT_SIZES     = [x for x in range(100, 1001, 500)]
INPUT_NOISES    = [0.00, 0.05, 0.10, 0.15]
INPUT_NOISE_SIZE_COMBOS = [{'n_points': a, 'noise': b} for a in INPUT_SIZES for b in INPUT_NOISES]
INPUT_NUM_MOONS = [3, 4, 5, 6]


# where to store current run results
RESULTS_DUMP_FOLDER     = f'./results/res_{datetime.datetime.now().strftime("%Y_%m_%d_T%H_%M_%S")}'
RESULTS_TIMING_DOC      = f'{RESULTS_DUMP_FOLDER}/time.csv'
RESULTS_CORRECTNESS_DOC = f'{RESULTS_DUMP_FOLDER}/correctness.csv'

# complete set of modules available to test for SpectralClustering

def pytest_configure(config):
    # make sure folders exist
    os.makedirs(os.path.dirname(RESULTS_TIMING_DOC     ), exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CORRECTNESS_DOC), exist_ok=True)

    # create empty files for both
    with open(RESULTS_TIMING_DOC,'w') as f:
        f.write('n_points,noise,variant,method,time\n')
    with open(RESULTS_CORRECTNESS_DOC,'w') as f:
        f.write('n_points,noise,variant,method,rand_index\n')

# generate the two moons problem
def binary_moons_data(n_points, noise):
    X, labels = sklearn_make_moons(n_points, noise, RAND_SEED)
    return X, labels

# TODO: helper functions: dumping new timing results, correctness etc.
def dump_time(n_points, noise, time, variant = 'DEFAULT', method = 'DEFAULT'):
    with open(RESULTS_TIMING_DOC,'a') as f:
        new_line = f'{n_points},{noise},{variant},{method},{time}\n'
        f.write(new_line)