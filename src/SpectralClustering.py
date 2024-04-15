from sklearn.base import ClusterMixin
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.pipeline_transformers import (
    NullTransformer                     ,
    affinity        as affinity_lib     ,
    refinement      as refinement_lib   ,
    laplacian       as laplacian_lib    ,
    decomposition   as decomposition_lib,
    embedding       as embedding_lib    ,
    clustering      as clustering_lib   ,
    confidence      as confidence_lib   ,
)

class SpectralClustering(ClusterMixin):

    DEFAULT_EPS = 0.4
    DEFAULT_K   = 20

    # TODO: add support for providing k/eps for NN graph generation, checking that they are provided when selecting appropriate methods

    # supported options for each pipeline component, note declared private to prevent mutation
    # TODO: add in supported methods for each part of pipeline
    COMPONENT_OPTIONS = {
        # data preprocessing: none, standard, min-max
        'standardisation': {
            'none'    : NullTransformer.NullTransformer(),
            'standard': StandardScaler(),
            'min-max' : MinMaxScaler(),
        },
        # similarity metrics to generate affinity matrix: euclidean, manhattan, Gaussian kernel
        'affinity': {
            'euclidean': affinity_lib.AffinityTransformer('euclidean'),
            'manhattan': affinity_lib.AffinityTransformer('manhattan'),
        },
        # graph refinement/connecting: complete, eps-radius, k-NN, mutual k-NN
        'refinement': {
            'eps'        : refinement_lib.EpsilonNNTransformer(DEFAULT_EPS),
            'knn'        : refinement_lib.kNNTransformer(DEFAULT_K),
            'mutual_knn' : refinement_lib.MutualKNNTransformer(DEFAULT_K),
            'none'       : refinement_lib.CompleteTransformer(),
        },
        # type of laplacian generated: standard, normalised 
        'laplacian': {
            'standard'  : laplacian_lib.LaplacianTransformer(normalize = False),
            'normalised': laplacian_lib.LaplacianTransformer(normalize = True ),
        },
        # method of eigendcomposition: standard dense, sparse improvements, specialised for Fiedler, Fourier transformations
        'decomposition': {
            'dense'      : decomposition_lib.DecompositionTransformer(method = 'dense'),
            'dense_eigh' : decomposition_lib.DecompositionTransformer(method = 'dense_eigh'),
            'sparse'     : decomposition_lib.DecompositionTransformer(method = 'sparse'),
            'sparse_eigh': decomposition_lib.DecompositionTransformer(method = 'sparse_eigh'),
        },
        # dimensionality of spectral embedding: single, more than one vec, dynamic selection of num_clusters
        'embedding': {
            'single': embedding_lib.EmbeddingTransformer(method = 'single'),
        },
        # method for post-clustering: k-means, agglomerative, DBScan etc.
        'clustering': {
            'k-means': clustering_lib.ClusteringTransformer(method = 'k-means', num_clusters = 2),
        },
        # whether to provide measure of confidence: True, False
        'confidence': {
            'false': NullTransformer.NullTransformer(),
        }
    }

    # TODO: add random state intialisation
    def __init__(
        self                         , num_clusters,
        standardisation = 'none'     , affinity   = 'euclidean',
        refinement      = 'eps'      , laplacian  = 'standard',
        decomposition   = 'dense'    , embedding  = 'single',
        clustering      = 'k-means'  , confidence = 'false',
        eps             = DEFAULT_EPS, k          = DEFAULT_K
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

        # TODO: check combination of parameters provided is valid
        # check if using eps refinement, eps param is valid
        # check if using k refinement, k is good
        self.COMPONENT_OPTIONS['refinement']['eps']        = refinement_lib.EpsilonNNTransformer(eps)
        self.COMPONENT_OPTIONS['refinement']['knn']        = refinement_lib.kNNTransformer(k)
        self.COMPONENT_OPTIONS['refinement']['mutual_knn'] = refinement_lib.MutualKNNTransformer(k)


        # TODO: build out pipeline (instead of if/else statements in fit)
        pipeline_steps = [
            ('standardisation', self.COMPONENT_OPTIONS['standardisation'][standardisation]),
            ('affinity'       , self.COMPONENT_OPTIONS['affinity'       ][affinity       ]),
            ('refinement'     , self.COMPONENT_OPTIONS['refinement'     ][refinement     ]),
            ('laplacian'      , self.COMPONENT_OPTIONS['laplacian'      ][laplacian      ]),
            ('decomposition'  , self.COMPONENT_OPTIONS['decomposition'  ][decomposition  ]),
            ('embedding'      , self.COMPONENT_OPTIONS['embedding'      ][embedding      ]),
            ('clustering'     , self.COMPONENT_OPTIONS['clustering'     ][clustering     ]),
            ('confidence'     , self.COMPONENT_OPTIONS['confidence'     ][confidence     ]),
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