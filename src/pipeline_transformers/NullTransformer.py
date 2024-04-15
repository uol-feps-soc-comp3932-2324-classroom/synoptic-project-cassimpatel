from sklearn.base import BaseEstimator, TransformerMixin

# for the purposes of transformations that don't do anything, a transformer that doesn't change it's inputs

class NullTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # don't do anything
        return X