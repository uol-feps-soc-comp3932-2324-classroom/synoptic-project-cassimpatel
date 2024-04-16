# COMP3932: Synoptic Project Repository

<!-- ABOUT THE PROJECT -->
## About The Project

This project is an investigation into a technique for clustering that uses the Graph Laplacian operator: Spectral Clustering. Implemented is a modular, customisable, packaged pipeline for this machine learning technique, a benchmarking framework that gathers results on runtime and performance, and a set of notebooks that demonstrate initial research and analysis into how this technique works, and its comparison to alternative clustering algorithms.  

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

To use the package and run benchmarking you will require the following prerequisites:

- Python 3.X
- Installation of `Make`, such that you can run Makefile commands


### Installation

To install the custom python packaged solution to your local package manager, follow these instructions:

1. Clone the repo
   ```sh
   git clone https://github.com/uol-feps-soc-comp3932-2324-classroom/synoptic-project-cassimpatel.git;
   ```
2. Create a virtual environment
   ``` sh
    python3 -m venv venv;
    source venv/bin/activate;
   ```
3. Install requirements (including source package)
   ```sh
   pip install -r requirements.txt;
   ```

<!-- USAGE EXAMPLES -->
## Usage

To use the package itself, you may import it like any other Python package:
```Python
from src.SpectralClustering    import SpectralClustering
from src.data_generation       import make_many_moons
from src.pipeline_transformers import (
    NullTransformer,
    affinity       ,
    refinement     ,
    laplacian      ,
    decomposition  ,
    embedding      ,
    clustering     ,
    confidence     ,
)
```
See examples of this usage within the analysis notebooks in `prelim_analysis/`. For details on parameters provided, see the source code.

## Testing & Benchmarking

To run the full testing framework use the following command:
```sh
make test;
```
This command will gather all tests, run them, and output results to a log file you will find in `tests/results/` as tests complete.

<!-- CONTACT -->
## Contact

Cassim Patel - 201394797 - sc20cp@leeds.ac.uk
