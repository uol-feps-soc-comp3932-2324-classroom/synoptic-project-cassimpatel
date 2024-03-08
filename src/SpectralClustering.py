from sklearn.base import ClusterMixin
from sklearn.pipeline import Pipeline

from src.pipeline_transformers import (
    NullTransformer,
    affinity,
    refinement,
    laplacian,
    decomposition,
    embedding,
    clustering,
    confidence,
)


class SpectralClustering(ClusterMixin):

    __EPS = 0.3
    __k_NN = 5
    # TODO: add support for providing k/eps for NN graph generation, checking that they are provided when selecting appropriate methods

    # supported options for each pipeline component, note declared private to prevent mutation
    # TODO: add in supported methods for each part of pipeline
    COMPONENT_OPTIONS = {
        # data preprocessing: none, z-score, min-max
        'standardisation': {
            'none': NullTransformer.NullTransformer(),
        },
        # similarity metrics to generate affinity matrix: euclidean, manhattan, Gaussian kernel
        'affinity': {
            'euclidean': affinity.AffinityTransformer('euclidean'),
            'manhattan': affinity.AffinityTransformer('manhattan'),
        },
        # graph refinement/connecting: complete, eps-radius, k-NN, mutual k-NN
        'refinement': {
            'eps': refinement.EpsilonNNTransformer(0.4),
        },
        # type of laplacian generated: standard, normalised 
        'laplacian': {
            'standard'  : laplacian.LaplacianTransformer(normalize = False),
            'normalised': laplacian.LaplacianTransformer(normalize = True ),
        },
        # method of eigendcomposition: standard dense, sparse improvements, specialised for Fiedler, Fourier transformations
        'decomposition': {
            'dense'      : decomposition.DecompositionTransformer(method = 'dense'),
            'dense_eigh' : decomposition.DecompositionTransformer(method = 'dense_eigh'),
            'sparse'     : decomposition.DecompositionTransformer(method = 'sparse'),
            'sparse_eigh': decomposition.DecompositionTransformer(method = 'sparse_eigh'),
        },
        # dimensionality of spectral embedding: single, more than one vec, dynamic selection of num_clusters
        'embedding': {
            'single': embedding.EmbeddingTransformer(method = 'single'),
        },
        # method for post-clustering: k-means, agglomerative, DBScan etc.
        'clustering': {
            'k-means': clustering.ClusteringTransformer(method = 'k-means', num_clusters = 2),
        },
        # whether to provide measure of confidence: True, False
        'confidence': {
            'false': NullTransformer.NullTransformer(),
        }
    }

    # TODO: add random state intialisation
    def __init__(
        self                       , num_clusters,
        standardisation = 'none'   , affinity   = 'euclidean',
        refinement      = 'eps'    , laplacian  = 'standard',
        decomposition   = 'dense'  , embedding  = 'single',
        clustering      = 'k-means', confidence = 'false'
    ):
        super().__init__()

        # check valid parameters are provided
        varname_display_pairs = [
            ('standardisation', standardisation),
            ('affinity'       , affinity       ),
            ('refinement'     , refinement     ),
            ('laplacian'      , laplacian      ),
            ('decomposition'  , decomposition  ),
            ('embedding'      , embedding      ),
            ('clustering'     , clustering     ),
            ('confidence'     , confidence     ),
        ]
        for (var, val) in varname_display_pairs:
            val_options = SpectralClustering.COMPONENT_OPTIONS[var].keys()
            if val not in val_options:
                raise ValueError(f"Parameter `{var}` must be one of {list(val_options)}")
            setattr(self, var, val)

        # check combination of parameters provided is valid

        # TODO: build out pipeline (instead of if/else statements in fit)
        pipeline_steps = [
            ('standardisation', SpectralClustering.COMPONENT_OPTIONS['standardisation'][standardisation]),
            ('affinity'       , SpectralClustering.COMPONENT_OPTIONS['affinity'       ][affinity       ]),
            ('refinement'     , SpectralClustering.COMPONENT_OPTIONS['refinement'     ][refinement     ]),
            ('laplacian'      , SpectralClustering.COMPONENT_OPTIONS['laplacian'      ][laplacian      ]),
            ('decomposition'  , SpectralClustering.COMPONENT_OPTIONS['decomposition'  ][decomposition  ]),
            ('embedding'      , SpectralClustering.COMPONENT_OPTIONS['embedding'      ][embedding      ]),
            ('clustering'     , SpectralClustering.COMPONENT_OPTIONS['clustering'     ][clustering     ]),
            ('confidence'     , SpectralClustering.COMPONENT_OPTIONS['confidence'     ][confidence     ]),
        ]
        self.pipeline = Pipeline(pipeline_steps)

    # TODO: provide 
    def fit(self, X):
        # TODO: check X type, shape of X, save expected shape for future
        shape = X.shape

        self.labels_ = self.pipeline.fit_transform(X)
        return self.labels_

    def predict(self, X):
        if self.confidence != 'k-means':
            raise ValueError('Cannot predict unseen points on model not trained with k-means post-clustering')
        
        raise ValueError('Predict method has not yet been implemented')
    
        # check X fits expected shape

        # map new points to same low dimensional space as fitted data

        # use post-clustering model to predict new classes in that space

        return X