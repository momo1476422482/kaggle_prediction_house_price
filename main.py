from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from dataset import house_price_features
from model import house_price_model
from typing import Callable


class house_price_predicter:
    def __init__(self, model: Callable) -> None:
        self.df_result = pd.DataFrame()
        self.model = model

    # =====================================================================
    def train_model(self, features: pd.DataFrame, reference: pd.DataFrame) -> None:
        self.model.gridsearch.fit(features, reference)
        print('best parameter', self.model.gridsearch.best_params_, 'best score', np.sqrt(-self.model.gridsearch.best_score_))

    # =====================================================
    def get_features_importance(self):
        features_importance = pd.DataFrame()
        features_importance['name'] = self.features.columns
        features_importance['importance'] = self.gridsearch.best_estimator_.coef_.reshape(-1, 1)
        self.features_importance = features_importance.sort_values('importance')

    # =====================================================
    def save_result(self, path_test: Path, features_test: pd.DataFrame) -> None:
        df_result = pd.read_csv(path_test)
        df_result['SalePrice'] = self.model(features_test)
        df_result = df_result[['Id', 'SalePrice']]
        df_result.to_csv(Path(__file__).parent / 'result.csv', index=False)

    # ===============================================
    def __call__(self, features_train: np.ndarray, reference_train: np.ndarray,
                 features_test: pd.DataFrame):
        self.train_model(features_train[0], reference_train)
        self.save_result(Path(__file__).parent / 'test.csv', features_test)


# =====================================================================================================
if __name__ == '__main__':
    features_train, reference_train, features_test = house_price_features(Path(__file__).parent / 'train.csv',
                                                                          Path(
                                                                              __file__).parent / 'test.csv').get_features_train_test(
        threshold=0, number_category=40)

    hpm = house_price_model('svr',param_grid={'C': [5,7,8,10],'gamma':[0.1,0.01,0.001]})

    hp = house_price_predicter(model=hpm)
    hp(features_train=features_train, features_test=features_test, reference_train=reference_train)
