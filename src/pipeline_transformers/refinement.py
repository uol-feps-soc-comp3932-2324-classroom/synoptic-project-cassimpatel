from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# TODO: add error checking

class EpsilonNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, eps):
        self.eps = eps
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.array(X < self.eps, dtype=np.float64)
        np.fill_diagonal(X,0)
    
        return X

class kNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.k = k
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n = len(X)

        idx = np.argpartition(X, self.k + 1, axis=0)
        idx = idx[:self.k+1, :]
        
        res = np.zeros((n, n))
        for i in range(n):
            idx_col = idx[:,i]
            res[idx_col,i] = 1

        np.fill_diagonal(res, 0)
        return res


class CompleteTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n = len(X)

        res = np.ones((n, n))
        np.fill_diagonal(res, 0)

        return res