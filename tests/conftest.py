import os
import time
import pytest
import signal
import datetime
from src.data_generation import sklearn_make_moons

# environment vars for preventing long runtime and repeats
NUM_REPEATS      = 3
MAX_TIMEOUT_SECS = 60 * 60

# input size/noise/num_clusters
RAND_SEED       = None
# INPUT_SIZES     = [x for x in range(100, 3001, 100)]
INPUT_SIZES     = [x for x in range(100, 500, 100)]
INPUT_NOISES    = [0.00, 0.05, 0.10, 0.15]
INPUT_NUM_MOONS = [3, 4, 5, 6]

# where to store current run results
RESULTS_DUMP_FOLDER     = f'./results/res_{datetime.datetime.now().strftime("%Y_%m_%d_T%H_%M_%S")}'
RESULTS_TIMING_DOC      = f'{RESULTS_DUMP_FOLDER}/results_times.csv'
RESULTS_CORRECTNESS_DOC = f'{RESULTS_DUMP_FOLDER}/results_correctness.csv'
RESULTS_REPORT_DOC      = f'{RESULTS_DUMP_FOLDER}/report.html'

# complete set of modules available to test for SpectralClustering



# setup before a testing session: make sure dump folders exist for results
def pytest_configure(config):
    # make sure folders exist
    os.makedirs(os.path.dirname(RESULTS_TIMING_DOC     ), exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CORRECTNESS_DOC), exist_ok=True)

    # create empty files for both
    with open(RESULTS_TIMING_DOC,'w') as f:
        f.write('n_points,noise,variant,method,time\n')
    with open(RESULTS_CORRECTNESS_DOC,'w') as f:
        f.write('n_points,noise,variant,method,adjusted_rand_index\n')

    # set creation of HTML report
    config.option.htmlpath = f'{RESULTS_DUMP_FOLDER}/report.html'
    config.option.self_contained_html = True


# generic handler to time a function call and capture result
def run_timeout_fn(fn, *args, **kwargs):
    # start a signal alarm to stop the function if it runs too long
    def handler(sig, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(MAX_TIMEOUT_SECS)

    # run the function with supplied arguments, capture result value
    try:
        start   = time.perf_counter()
        results = fn(*args, **kwargs)
        end     = time.perf_counter()
        signal.alarm(0)
    except TimeoutError:
        return (MAX_TIMEOUT_SECS, 'TIMEOUT')
    
    exec_time = end - start
    return (exec_time, results)


# generate the two moons problem
def binary_moons_data(n_points, noise):
    X, labels = sklearn_make_moons(n_points, noise, RAND_SEED)
    return X, labels

# helper functions: dumping new timing results, correctness etc.
def dump_time(n_points, noise, time, variant = 'DEFAULT', method = 'DEFAULT'):
    with open(RESULTS_TIMING_DOC,'a') as f:
        new_line = f'{n_points},{noise},{variant},{method},{time}\n'
        f.write(new_line)
def dump_correctness(n_points, noise, adj_rand_index, variant = 'DEFAULT', method = 'DEFAULT'):
    with open(RESULTS_CORRECTNESS_DOC,'a') as f:
        new_line = f'{n_points},{noise},{variant},{method},{adj_rand_index}\n'
        f.write(new_line)