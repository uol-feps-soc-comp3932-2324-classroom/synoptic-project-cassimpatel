import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin
from sklearn.pipeline import Pipeline

from src.pipeline_transformers import (
    affinity
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
            'none': None,
        },
        'affinity': {
            # similarity metrics to generate affinity matrix: euclidean, manhattan, Gaussian kernel
            'euclidean': affinity.AffinityTransformer('euclidean'),
        },
        'refinement': {
            # graph refinement/connecting: complete, eps-radius, k-NN, mutual k-NN
            'eps': None,
        },
        'laplacian': {
            # type of laplacian generated: standard, normalised 
            'standard': None,
        },
        'decomposition': {
            # method of eigendcomposition: standard dense, sparse improvements, specialised for Fiedler, Fourier transformations
            'dense': None,
        },
        'embedding': {
            # dimensionality of spectral embedding: single, more than one vec, dynamic selection of num_clusters
            'single': None,
        },
        'clustering': {
            # method for post-clustering: k-means, agglomerative, DBScan etc.
            'k-means': None,
        },
        'confidence': {
            # whether to provide measure of confidence: True, False
            'false': None,
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
            # ('refinement'     , SpectralClustering.__COMPONENT_OPTIONS['refinement'     ][refinement     ]),
            # ('laplacian'      , SpectralClustering.__COMPONENT_OPTIONS['laplacian'      ][laplacian      ]),
            # ('decomposition'  , SpectralClustering.__COMPONENT_OPTIONS['decomposition'  ][decomposition  ]),
            # ('embedding'      , SpectralClustering.__COMPONENT_OPTIONS['embedding'      ][embedding      ]),
            # ('clustering'     , SpectralClustering.__COMPONENT_OPTIONS['clustering'     ][clustering     ]),
            # ('confidence'     , SpectralClustering.__COMPONENT_OPTIONS['confidence'     ][confidence     ]),
        ]
        self.pipeline = Pipeline(pipeline_steps)

    # TODO: provide 
    def fit(self, X):
        x = self.pipeline.fit_transform(X)
        print(X.shape, x.shape)
        return


        # TODO: check X type, shape of X, save expected shape for future
        shape = X.shape

        # step 1: standardisation
        if self.standardisation == 'none':
            pass
        else:
            raise ValueError(f"Required module parameter has not yet been implemented")

        # step 2: affinity
        if self.affinity == 'euclidean':
            A = pairwise_distances(X, X, 'euclidean')
            # print(A)
            pass
        else:
            pass
            

        # step 3: refinement
        if self.refinement == 'eps':
            # TODO: remove hard-coded eps param
            A[A < 0.4] = 1
            A[A!= 1] = 0
            np.fill_diagonal(A,0)

            n = len(X)
            D = np.zeros((n, n))
            d = [np.sum(A[row,:]) for row in range(A.shape[0])]
            np.fill_diagonal(D, d)

            # print('A\n', A, 'D\n', D)
            pass
        else:
            raise ValueError(f"Required module parameter has not yet been implemented")

        # step 4: normalisation
        if self.laplacian == 'standard':
            L = D - A
            # print('L\n', L)
            pass
        else:
            raise ValueError(f"Required module parameter has not yet been implemented")

        # step 5: decomposition
        if self.decomposition == 'dense':
            eig_val, eig_vec = np.linalg.eig(L)
            eig_val = eig_val.real
            eig_vec = eig_vec.real
            
            pass
        else:
            raise ValueError(f"Required module parameter has not yet been implemented")

        # step 6: embedding
        if self.embedding == 'single':
            eig_val  = eig_val.argsort()
            z_eigvec = eig_vec[:,eig_val][:,1]

            # print('z\n', z_eigvec)
            # z_eigvec[z_eigvec >= 0] = 1
            # z_eigvec[z_eigvec < 0] = 0
            pass
        else:
            raise ValueError(f"Required module parameter has not yet been implemented")

        # step 7: clustering
        if self.clustering == 'k-means':
            real_values = z_eigvec.reshape(-1, 1)
            # train a KMeans Clustering Model on the Fiedler eigenvector
            kmeans_model = KMeans(n_clusters=2).fit(real_values)
            self.labels_ = kmeans_model.labels_
            return self.labels_
            pass
        else:
            raise ValueError(f"Required module parameter has not yet been implemented")

        # step 8: confidence
        if self.confidence == 'false':
            pass
        else:
            raise ValueError(f"Required module parameter has not yet been implemented")

        # run pipeline

        # set results to self.labels_

        return self

    def predict(self, X):
        if self.confidence != 'k-means':
            raise ValueError('Cannot predict unseen points on model not trained with k-means post-clustering')
        
        raise ValueError('Predict method has not yet been implemented')
    
        # check X fits expected shape

        # map new points to same low dimensional space as fitted data

        # use post-clustering model to predict new classes in that space

        return X