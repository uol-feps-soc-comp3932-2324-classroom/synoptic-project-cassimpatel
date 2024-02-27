[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-7f7980b617ed060a017424585567c406b6ee15c891e84e1186181d67ecf80aa0.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=12985138)


# Requirements / Architecture

Thinking around what will be needed as the full solution

## Misc
- input generator: two moons given n_points / noisiness
- additionally provide a best-solution
- EXTENSION: support for many interleaving moons given k (conveniently number of clusters)
    - each moon gets n/k points
    - each moon generated individually, then shift and flip depending on stride
    - generate ground truth as the index in num moons

## Overall solution

- `_SpectralClustering` transformer
    - includes pipeline
    - init defines the pipeline taken, hence pipeline is an attribute
    - default options available for all options of pipeline
    - Issue:
        - how does this work for only transforming test data? calculate similarity to other points, recalculate eigdecomp???
    - will require custom transformers for some subparts, others are prebuilt e.g. k-means
- benchmarking system
    - a combinatorial solution
    - should be able to leave to run for long amounts of time
    - method to exhaustively test single sub-module of spectral clustering solution
    - options: problem_size/noisiness/method used, additionally num_times to re-run (for averaging), max_time (to kill once complete)
    - dumping solutions as JSON for protection
    - CONSIDERATION: built in options for correctness calculations???
        - take a solution and calculate accuracy of clustering - and additional metrics around classification
        - offer other (clustering based) metrics??
            - see this https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    - numerical stability calculations
        - cannot test the pipeline as a whole - will need to test individual parts, with underlying ground truth???
        - alternatively - use theory and do by hand

## Additional solutions required 
- will need to fill this in later
- some pipeline components already exist e.g. k-means, scalars, etc.
- others will need to be created, program custom fit solution
- make use of plethera of similarity measures
    - https://scikit-learn.org/stable/modules/metrics.html

## Fallback plans
- Risk: cannot implement everything in time
- Solution: create default simple versions of each module for PoC e.g. k-means for final clustering etc.
- then add extra options and test for results as you go


