
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        return X