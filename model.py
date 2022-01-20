from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA

class house_price_predicter:
    def __init__(self, path_csv: Path, path_test_csv: Path) -> None:
        self.df_result = pd.DataFrame()
        self.model = None
        self.df_train = pd.read_csv(path_csv)
        self.df_test = pd.read_csv(path_test_csv)
        self.df = pd.concat([self.df_train, self.df_test])

    # =====================================================
    def get_basic_info(self, df: pd.DataFrame) -> None:
        plt.figure()
        sns.distplot(np.log(self.df['SalePrice'].to_numpy()))
        plt.savefig('hist.png')

        list_features = list(df.columns)
        self.list_numerical_features = list(df.corr().index)
        for feature_nu in self.list_numerical_features:
            list_features.remove((feature_nu))
        self.list_category_features = list_features

        print('categorical nan', self.df[self.list_category_features].isna().sum())
        print('numerical nan', self.df[self.list_numerical_features].isna().sum())

    # =========================================================
    def plot_data(self, df: pd.DataFrame, feature: str) -> None:
        plt.figure()
        sns.barplot(x=feature, y='SalePrice', data=df)
        plt.savefig(f'{feature}_bar.png')

        fig, ax = plt.subplots(figsize=(24, 10))
        s = sns.barplot(x='category', y='pvalues', data=self.df_category)
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
        plt.savefig(Path(__file__).parent / 'category.png')

        plt.figure()
        plt.subplots(figsize=(32, 20))
        sns.barplot(x='importance', y='name', data=self.features_importance)
        plt.savefig('feature_imporantce.png')
        print('feature importance :', self.features_importance.sort_values('importance').to_string())

    # ==========================================
    def select_numerical_features(self, df: pd.DataFrame, threshold: float) -> List[str]:
        coor_matrix = df.corr()
        list_numerical_features = list(
            coor_matrix.loc[
                (coor_matrix['SalePrice'] > threshold) | (coor_matrix['SalePrice'] < -threshold), 'SalePrice'].index)
        list_numerical_features.remove('SalePrice')
        return list_numerical_features

    # ====================================================================================
    def select_categorical_features(self, df: pd.DataFrame, number: int) -> List[str]:
        assert number <= 40, 'number should be <=40'

        def anova(list_category_features: List[str]) -> pd.DataFrame:
            pvalues = []
            df_category = pd.DataFrame()
            df_category['category'] = list_category_features

            for category in list_category_features:
                list_category_values = list(df[category].unique())
                samples = []
                for value in list_category_values:
                    samples.append(df.loc[(df[category] == value), 'SalePrice'].values)
                pvalues.append(np.log(1 / (stats.f_oneway(*samples)[1])))

            df_category['pvalues'] = pvalues
            df_category = df_category.sort_values('pvalues')
            return df_category

        self.df_category = anova(self.list_category_features)
        list_features_cat = list(self.df_category['category'][::-1])
        return list_features_cat[0:number + 1]

    # ==========================================
    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Numerical features
        cols = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea",'GarageYrBlt','BsmtFullBath','LotFrontage','BsmtHalfBath']
        for col in cols:
            df[col].fillna(0, inplace=True)

        # Categorical Features
        for cat in self.list_category_features:
            df.loc[df[cat].isna(), cat] = df[cat].value_counts().index[0]
        return df

    # ==========================================
    def drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df=df.drop(columns=['GarageArea', '1stFlrSF'])
        return df

    # ==========================================
    def get_encodage_categorical_features(self, df: pd.DataFrame, cat: str) -> np.ndarray:

        list_category_values = list(df[cat].unique())
        samples = []
        for value in list_category_values:
            samples.append(df.loc[(df[cat] == value), 'SalePrice'].values.mean())
        return np.array(samples).argsort()

    # ==========================================
    def transform_catecorical_features(self, df: pd.DataFrame,
                                       list_cat_feautres: List[str]) -> pd.DataFrame:
        for category in list_cat_feautres:
            order = self.get_encodage_categorical_features(df.iloc[0:self.df_train.shape[0], :], category)
            list_category_values = list(df[category].unique())
            for index, i in enumerate(order):
                df.loc[(df[category] == list_category_values[i]), category] = index
        return df

    # ==================================================
    def transform_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = df.mean()
        sigma = df.std()
        return (df - mean) / sigma

    # ==================================================
    def transform_total_features(self,df:pd.DataFrame)->pd.DataFrame:
        df["TotalHouse"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["TotalArea"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"] +df["GarageArea"]
        df["+_TotalHouse_OverallQual"] = df["TotalHouse"] * df["OverallQual"]
        return df

    # ==========================================
    def get_features(self, threshold: float, number_category: int) -> None:
        df = self.impute_missing_values(self.df)

        list_features_numerical = self.select_numerical_features(df.iloc[0:self.df_train.shape[0], :], threshold)
        list_features_catecorical = self.select_categorical_features(df.iloc[0:self.df_train.shape[0], :],
                                                                     number_category)
        df_categori = self.transform_catecorical_features(df, list_features_catecorical)
        df_categori = df_categori[list_features_catecorical].astype(float)
        df_numerical = self.transform_numerical_features(df[list_features_numerical])
        self.features = pd.concat((df_numerical, df_categori), axis=1)
        self.features=self.transform_total_features(self.features)

    # ==========================================
    def train_model(self, algo: str, features: np.ndarray, reference: np.ndarray, param_grid: Dict) -> None:

        if algo == 'Ridge':
            ridge = Ridge(alpha=15)
            self.model = ridge
        if algo == 'svr':
            svr = SVR(gamma=0.001, kernel='rbf', C=30, epsilon=0.1)
            self.model = svr

        if algo == 'krr':
            # kernel ridge
            krr = KernelRidge(alpha=0.5,kernel='polynomial',degree=3,coef0=0.5)
            self.model = krr

        gridsearch = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error')
        gridsearch.fit(features, reference)
        print('best parameter', gridsearch.best_params_, 'best score', np.sqrt(-gridsearch.best_score_))
        self.gridsearch = gridsearch

    # =====================================================
    def get_features_importance(self):
        features_importance = pd.DataFrame()
        features_importance['name'] = self.features.columns
        features_importance['importance'] = self.gridsearch.best_estimator_.coef_.reshape(-1,1)
        self.features_importance=features_importance.sort_values('importance')

    # =====================================================
    def predict_price(self, algo: str, param_grid: Dict) -> np.ndarray:
        self.train_model(algo=algo, features=self.features.iloc[0:self.df_train.shape[0], :],
                         reference=np.reshape(np.log(self.df_train["SalePrice"].to_numpy()),
                                              (-1, 1)), param_grid=param_grid)

        X_test = self.features.iloc[self.df_train.shape[0]:, :]

        return np.exp(self.gridsearch.predict(X_test))

    # =====================================================
    def save_result(self, algo: str, param_grid: Dict) -> None:
        self.df_result = self.df_test
        self.df_result['SalePrice'] = self.predict_price(algo, param_grid)
        self.df_result = self.df_result[['Id', 'SalePrice']]
        self.df_result.to_csv(Path(__file__).parent / 'result.csv', index=False)

    # ===============================================
    def __call__(self, algo: str, param_grid: Dict, threshold_feature_numerical: float=0.5, num_feature_category: int=25):
        self.get_basic_info(self.df)
        self.get_features(threshold_feature_numerical,num_feature_category)
        self.save_result(algo, param_grid)
        #self.get_features_importance()


# =====================================================================================================
if __name__ == '__main__':
    hp = house_price_predicter(Path(__file__).parent / 'train.csv', Path(__file__).parent / 'test.csv')
    hp('svr', param_grid={'C': [140,150], 'gamma': [0.1, 0.01, 0.001, 0.0001] },threshold_feature_numerical=0,num_feature_category=36)
    #hp('krr', param_grid={'alpha': [0.001], 'degree': [1] ,'coef0':[0.1,0.2,0.5,0.7,0.8]})

