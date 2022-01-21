from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from typing import Dict


class house_price_model:
    def __init__(self,algo:str,param_grid:Dict) -> None:

        self.algo = algo
        if self.algo == 'Ridge':
            m = Ridge(alpha=15)

        elif self.algo == 'svr':
            m = SVR(gamma=0.001, kernel='rbf', C=30, epsilon=0.1)

        elif self.algo == 'krr':
            # kernel ridge
            m = KernelRidge(alpha=0.5, kernel='polynomial', degree=3, coef0=0.5)
        self.model=m
        self.gridsearch = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error')

    # ==========================================
    def __call__(self,features_test:pd.DataFrame) -> np.ndarray:
        return np.exp(self.gridsearch.predict(features_test))







