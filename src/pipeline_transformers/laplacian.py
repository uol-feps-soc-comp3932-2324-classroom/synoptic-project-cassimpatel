from sklearn.base import BaseEstimator, TransformerMixin

# TODO: add error checking

class LaplacianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method = None):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X