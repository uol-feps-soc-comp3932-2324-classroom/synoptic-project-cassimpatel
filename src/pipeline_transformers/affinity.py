from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances

# TODO: add error checking

class AffinityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method = 'euclidean'):
        if method != 'euclidean':
            raise ValueError(f"Required module parameter has not yet been implemented")
        self.method = method
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pairwise_distances(X, X, self.method)