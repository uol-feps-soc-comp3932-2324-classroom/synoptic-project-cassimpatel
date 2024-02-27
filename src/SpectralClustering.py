
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.pipeline import Pipeline

class SpectralClustering(ClusterMixin):

    # supported options for each pipeline component, note declared private to prevent mutation
    # TODO: add in supported methods for each part of pipeline
    __COMPONENT_OPTIONS = {
        'standardisation': {
            # data preprocessing: none, z-score, min-max
        },
        'affinity': {
            # similarity metrics to generate affinity matrix: euclidean, manhattan, Gaussian kernel
        },
        'refinement': {
            # graph refinement/connecting: complete, eps-radius, k-NN, mutual k-NN
        },
        'normalisation': {
            # type of laplacian generated: standard, normalised 
        },
        'decomposition': {
            # method of eigendcomposition: standard dense, sparse improvements, specialised for Fiedler, Fourier transformations
        },
        'embedding': {
            # dimensionality of spectral embedding: single, more than one vec, dynamic selection of num_clusters
        },
        'clustering': {
            # method for post-clustering: k-means, agglomerative, DBScan etc.
        },
        'confidence': {
            # whether to provide measure of confidence: True, False
        }
    }

    def __init__(self, num_clusters, standardisation, affinity, refinement, normalisation, decomposition, embedding, clustering, confidence):
        # check valid parameters are provided

        # check combination of parameters provided is valid

        # build out pipeline

        return self

    # TODO: provide 
    def fit(self, X):
        # check shape of X, save expected shape for future

        # run pipeline

        # set results to self.labels_

        return self

    def predict(self, X):
        # check X fits expected shape

        # map new points to same low dimensional space as fitted data

        # use post-clustering model to predict new classes in that space

        return X