from pathlib import Path
from typing import List, Dict,Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class house_price_features:
    def __init__(self, path_csv: Path, path_test_csv: Path) -> None:

        self.df_train = pd.read_csv(path_csv)
        self.df_test = pd.read_csv(path_test_csv)
        self.df = pd.concat([self.df_train, self.df_test])

        self.list_numerical_features = list(self.df.corr().index)
        list_features = list(self.df.columns)
        for feature_nu in self.list_numerical_features:
            list_features.remove((feature_nu))
        self.list_category_features = list_features
    # =====================================================
    def get_basic_info(self, df: pd.DataFrame) -> None:
        """
        Get the basic information of the dataframe : the numerical features,categorical features and their respective nan values
        :param df:
        :return:
        """
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
        """
        Select the numerical features which are most correlated to the house price
        :param df:
        :param threshold: a float permitting to filter the numerical features
        :return:
        """
        coor_matrix = df.corr()
        list_numerical_features = list(
            coor_matrix.loc[
                (coor_matrix['SalePrice'] > threshold) | (coor_matrix['SalePrice'] < -threshold), 'SalePrice'].index)
        list_numerical_features.remove('SalePrice')
        return list_numerical_features

    # ====================================================================================
    def select_categorical_features(self, df: pd.DataFrame, number: int) -> List[str]:
        """
        Select the most influencial categorical features of House Price by one-way Anova
        :param df:
        :param number: Number of the categorical features to be remained
        :return:
        """
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
        cols = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea",
                'GarageYrBlt', 'BsmtFullBath', 'LotFrontage', 'BsmtHalfBath']
        for col in cols:
            df[col].fillna(0, inplace=True)

        # Categorical Features
        for cat in self.list_category_features:
            df.loc[df[cat].isna(), cat] = df[cat].value_counts().index[0]
        return df

    # ================================================================================
    def get_encodage_categorical_features(self, df: pd.DataFrame, cat: str) -> np.ndarray:
        """
        Get the encodage label of categorical features by the mean value of the corresponding house price
        :param df:
        :param cat:
        :return:
        """

        list_category_values = list(df[cat].unique())
        samples = []
        for value in list_category_values:
            samples.append(df.loc[(df[cat] == value), 'SalePrice'].values.mean())
        return np.array(samples).argsort()

    # =======================================================
    def transform_catecorical_features(self, df: pd.DataFrame,
                                       list_cat_feautres: List[str]) -> pd.DataFrame:
        for category in list_cat_feautres:
            order = self.get_encodage_categorical_features(df.iloc[0:self.df_train.shape[0], :], category)
            list_category_values = list(df[category].unique())
            for index, i in enumerate(order):
                df.loc[(df[category] == list_category_values[i]), category] = index
        return df

    # ===============================================================
    def transform_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = df.mean()
        sigma = df.std()
        return (df - mean) / sigma

    # =================================================================
    def transform_total_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get some new features by the fusion of the existing ones
        :param df:
        :return:
        """
        df["TotalHouse"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["TotalArea"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"] + df["GarageArea"]
        df["+_TotalHouse_OverallQual"] = df["TotalHouse"] * df["OverallQual"]
        return df

    # =========================================================================
    def get_features(self, threshold: float, number_category: int) -> pd.DataFrame:
        df = self.impute_missing_values(self.df)

        list_features_numerical = self.select_numerical_features(df.iloc[0:self.df_train.shape[0], :], threshold)
        list_features_catecorical = self.select_categorical_features(df.iloc[0:self.df_train.shape[0], :],
                                                                     number_category)
        df_categori = self.transform_catecorical_features(df, list_features_catecorical)
        df_categori = df_categori[list_features_catecorical].astype(float)
        df_numerical = self.transform_numerical_features(df[list_features_numerical])
        self.features = pd.concat((df_numerical, df_categori), axis=1)
        return self.transform_total_features(self.features)
    # =========================================================================

    def get_features_train_test(self, threshold: float=0, number_category: int=35) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        self.features=self.get_features(threshold, number_category)
        features_train = self.features.iloc[0:self.df_train.shape[0], :],
        reference_train = np.reshape(np.log(self.df_train["SalePrice"].to_numpy()), (-1, 1))
        features_test = self.features.iloc[self.df_train.shape[0]:, :]
        return features_train,reference_train,features_test
