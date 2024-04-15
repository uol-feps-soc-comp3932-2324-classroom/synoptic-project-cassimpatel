from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

# TODO: add error checking

class ClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method = 'k-means', num_clusters = 2):
        if method != 'k-means':
            raise ValueError(f"Required module parameter has not yet been implemented")

        self.method = method
        self.num_clusters = num_clusters
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # TODO: add normalisation??

        kmeans_model = KMeans(n_clusters=self.num_clusters).fit(X)
        return kmeans_model.labels_