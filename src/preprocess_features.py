import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import Config
from logger import AppLogger


class FeaturePreprocessor:
    """
    This class shall be used for preprocessing the features.
    """

    def __init__(self, file_object):
        self.file_object = file_object
        self.logger_object = AppLogger()

    def get_num_features(self, features_df):
        """
        This method is used for getting a list numerical features.
        :param features_df: features dataframe
        :return: list of numerical features
        """
        self.logger_object.log(
            self.file_object,
            "Entered get_num_features method of FeaturePreprocessor class.",
        )
        try:
            num_cols = features_df.select_dtypes(exclude="object").columns.to_list()
            self.logger_object.log(
                self.file_object, "Successfully got list of numerical features."
            )
            return num_cols
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in get_num_features method of FeaturePreprocessor class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Getting list of numerical features unsuccessful. Exited get_num_features method of FeaturePreprocessor class",
            )
            raise Exception()

    def get_cat_features(self, features_df):
        """
        This method is used for getting a list categorical features.
        :param features_df: features dataframe
        :return: list of categorical features
        """
        self.logger_object.log(
            self.file_object,
            "Entered get_cat_features method of FeaturePreprocessor class.",
        )
        try:
            cat_cols = features_df.select_dtypes(include="object").columns.to_list()
            self.logger_object.log(
                self.file_object, "Successfully returned list of categorical features."
            )
            return cat_cols
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in get_cat_features method of FeaturePreprocessor class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Getting list of categorical features unsuccessful. Exited get_cat_features method of FeaturePreprocessor class",
            )
            raise Exception()

    def preprocess_num_features(self, train_df, test_df, num_cols):
        """
        This method is used for preprocessing numerical features and saves as csv.
        :param features_df: features dataframe
        :return: None
        """
        self.logger_object.log(
            self.file_object,
            "Entered preprocess_num_features method of FeaturePreprocessor class.",
        )
        try:
            scaler = StandardScaler()
            scaler.fit(train_df[num_cols])
            train_df[num_cols] = scaler.transform(train_df[num_cols])
            test_df[num_cols] = scaler.transform(test_df[num_cols])
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in preprocess_num_features method of FeaturePreprocessor class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Preprocessing numerical features unsuccessful. Exited preprocess_num_features method of FeaturePreprocessor class",
            )
            raise Exception()

    def preprocess_cat_features(self, train_df, test_df, cat_cols):
        """
        This method is used for preprocessing categorical features and saves as csv.
        :param features_df: features dataframe
        :return: None
        """
        self.logger_object.log(
            self.file_object,
            "Entered preprocess_cat_features method of FeaturePreprocessor class.",
        )
        try:
            ohe = OneHotEncoder(sparse=False, drop="first")
            ohe.fit(train_df[cat_cols])
            encoded_cols = list(ohe.get_feature_names(cat_cols))
            train_df[encoded_cols] = ohe.transform(train_df[cat_cols])
            test_df[encoded_cols] = ohe.transform(test_df[cat_cols])
            self.logger_object.log(
                self.file_object, "Successfully preprocessd categorical columns."
            )
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in preprocess_cat_features method of FeaturePreprocessor class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Preprocessing categorical features unsuccessful. Exited preprocess_cat_features method of FeaturePreprocessor class",
            )
            raise Exception()

    def drop_features(self, features_df, drop_cols):
        """
        This method is used for dropping unnecessary columns.
        :param features_df: features dataframe
        :return: None
        """
        self.logger_object.log(
            self.file_object,
            "Entered drop_features method of FeaturePreprocessor class.",
        )
        try:
            cat_cols = self.get_cat_features(features_df)
            features_df.drop(columns=drop_cols, inplace=True)
            self.logger_object.log(self.file_object, "Successfully dropped features.")
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in drop_features method of FeaturePreprocessor class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Dropping features unsuccessful. Exited drop_features method of FeaturePreprocessor class",
            )
            raise Exception()

    def save_preprocessed_features(self, features_df, outfile_name):
        """
        This method is used for saving the preprocessed features to csv.
        :return:
        """
        self.logger_object.log(
            self.file_object,
            "Entered save_preprocessed_features method of FeaturePreprocessor class.",
        )
        try:
            features_df.to_csv(
                str(Config.FEATURES_PATH / f"{outfile_name}.csv"), index=None
            )
            self.logger_object.log(
                self.file_object, "Successfully saved preprocessed features."
            )
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in save_preprocessed_features method of FeaturePreprocessor class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Saving preprocessed features unsuccessful. Exited save_preprocessed_features method of FeaturePreprocessor class",
            )
            raise Exception()
