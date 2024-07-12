import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NewFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']
        X['GrLivArea'] = np.log(X['GrLivArea'])
        #X['BsmtUnfSF'] = np.log(X['BsmtUnfSF'])
        X['LotFrontage'] = np.log(X['LotFrontage'])
        #X['hasgarage'] = X['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        #X['MSSubClass'] = X['MSSubClass'].apply(str)
        #X['OverallCond'] = X['OverallCond'].astype(str)
        #X['YrSold'] = X['YrSold'].astype(str)
        #X['MoSold'] = X['MoSold'].astype(str)
        return X