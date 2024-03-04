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
    __COMPONENT_OPTIONS = {
        'standardisation': {
            # data preprocessing: none, z-score, min-max
            'none': NullTransformer.NullTransformer(),
        },
        'affinity': {
            # similarity metrics to generate affinity matrix: euclidean, manhattan, Gaussian kernel
            'euclidean': affinity.AffinityTransformer('euclidean'),
        },
        'refinement': {
            # graph refinement/connecting: complete, eps-radius, k-NN, mutual k-NN
            'eps': refinement.EpsilonNNTransformer(0.4),
        },
        'laplacian': {
            # type of laplacian generated: standard, normalised 
            'standard': laplacian.LaplacianTransformer(normalize=False),
        },
        'decomposition': {
            # method of eigendcomposition: standard dense, sparse improvements, specialised for Fiedler, Fourier transformations
            'dense': decomposition.DecompositionTransformer(method = 'dense'),
        },
        'embedding': {
            # dimensionality of spectral embedding: single, more than one vec, dynamic selection of num_clusters
            'single': embedding.EmbeddingTransformer(method = 'single'),
        },
        'clustering': {
            # method for post-clustering: k-means, agglomerative, DBScan etc.
            'k-means': clustering.ClusteringTransformer(method = 'k-means', num_clusters=2),
        },
        'confidence': {
            # whether to provide measure of confidence: True, False
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
            val_options = SpectralClustering.__COMPONENT_OPTIONS[var].keys()
            if val not in val_options:
                raise ValueError(f"Parameter `{var}` must be one of {list(val_options)}")

        # set parameters
        for (var, val) in varname_display_pairs:
            setattr(self, var, val)

        # check combination of parameters provided is valid

        # TODO: build out pipeline (instead of if/else statements in fit)
        pipeline_steps = [
            ('standardisation', SpectralClustering.__COMPONENT_OPTIONS['standardisation'][standardisation]),
            ('affinity'       , SpectralClustering.__COMPONENT_OPTIONS['affinity'       ][affinity       ]),
            ('refinement'     , SpectralClustering.__COMPONENT_OPTIONS['refinement'     ][refinement     ]),
            ('laplacian'      , SpectralClustering.__COMPONENT_OPTIONS['laplacian'      ][laplacian      ]),
            ('decomposition'  , SpectralClustering.__COMPONENT_OPTIONS['decomposition'  ][decomposition  ]),
            ('embedding'      , SpectralClustering.__COMPONENT_OPTIONS['embedding'      ][embedding      ]),
            ('clustering'     , SpectralClustering.__COMPONENT_OPTIONS['clustering'     ][clustering     ]),
            ('confidence'     , SpectralClustering.__COMPONENT_OPTIONS['confidence'     ][confidence     ]),
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