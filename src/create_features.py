import pandas as pd

from config import Config
from logger import AppLogger


class FeatureExtractor:
    """
    This class shall be used for extracting features from the original dataset.
    """

    def __init__(self, file_object):
        self.file_object = file_object
        self.logger_object = AppLogger()
        Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)
        self.train_df = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
        self.test_df = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))
        self.features_list = ["age", "sex", "bmi", "children", "smoker", "region"]
        self.labels_list = ["expenses"]

    def extract_features(self):
        """
        This method extracts features dataframe and saves  as csv.
        :param df: dataframe from which features are to be extracted
        :return: None
        """
        self.logger_object.log(
            self.file_object,
            "Entered extract_features method of FeatureExtractor class",
        )
        try:
            train_features_df = self.train_df[self.features_list]
            test_features_df = self.test_df[self.features_list]
            self.logger_object.log(
                self.file_object, "Train and test features read successfully."
            )
            train_features_df.to_csv(
                str(Config.FEATURES_PATH / "train_features.csv"), index=None
            )
            test_features_df.to_csv(
                str(Config.FEATURES_PATH / "test_features.csv"), index=None
            )
            self.logger_object.log(
                self.file_object,
                "Train and test features saved as train_features.csv and test_features.csv respectively and successfully.",
            )
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in extract_features method of FeatureExtractor class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Extract features unsuccessful. Exited extract_features method of FeatureExtractor class",
            )
            raise Exception()

    def extract_labels(self):
        """
        This method extracts labels from the dataframe and saves as csv.
        :param df: dataframe from which labels are to be extracted
        :return: labels dataframe
        """
        self.logger_object.log(
            self.file_object, "Entered extract_labels method of FeatureExtractor class"
        )
        try:
            train_labels_df = self.train_df[self.labels_list]
            test_labels_df = self.test_df[self.labels_list]
            self.logger_object.log(
                self.file_object, "Train and test features read successfully."
            )
            train_labels_df.to_csv(
                str(Config.FEATURES_PATH / "train_labels.csv"), index=None
            )
            test_labels_df.to_csv(
                str(Config.FEATURES_PATH / "test_labels.csv"), index=None
            )
            self.logger_object.log(
                self.file_object,
                "Train and test features saved as train_labels.csv and test_labels.csv respectively and successfully.",
            )
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in extract_labels method of FeatureExtractor class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Extract labels unsuccessful. Exited extract_features method of FeatureExtractor class",
            )
            raise Exception()
