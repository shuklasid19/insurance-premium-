import numpy as np
import opendatasets as od
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config
from logger import AppLogger


class DatasetFetcher:
    """
    This class shall be used to fetch the dataset from the source.
    """

    def __init__(self, file_object):
        self.file_object = file_object
        self.logger_object = AppLogger()
        np.random.seed(Config.RANDOM_SEED)
        Config.ORIGINAL_DATASET_FILE_PATH.parent.parent.mkdir(
            parents=True, exist_ok=True
        )
        Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

    def fetch_dataset(self):
        """
        This method fetches dataset from the url and saves as original_dataset.
        :return: None
        """
        self.logger_object.log(
            self.file_object, "Entered fetch_dataset method of DataFetcher class."
        )
        try:
            od.download(
                Config.DATASET_URL, str(Config.ORIGINAL_DATASET_FILE_PATH.parent.parent)
            )
            self.logger_object.log(self.file_object, "Dataset downloaded successfully.")
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in fetch_dataset method of DataFetcher class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Dataset download unsuccessful. Exited fetch_dataset method of DataFetcher class",
            )
            raise Exception()

    def split_dataset(self, test_size=0.2):
        """
        This method splits the original dataset into training and testing data and saves each of them as csv.
        :return: None
        """
        self.logger_object.log(
            self.file_object, "Entered split_dataset method of DataFetcher class."
        )
        try:
            df = pd.read_csv(str(Config.ORIGINAL_DATASET_FILE_PATH))
            self.logger_object.log(
                self.file_object, "Successfully read original_dataset."
            )
            df_train, df_test = train_test_split(
                df, test_size=test_size, random_state=Config.RANDOM_SEED
            )
            self.logger_object.log(
                self.file_object,
                f"Successfully split train and test data with test_size={test_size}.",
            )
            df_train.to_csv(str(Config.DATASET_PATH / "train.csv"), index=None)
            df_test.to_csv(str(Config.DATASET_PATH / "test.csv"), index=None)
            self.logger_object.log(
                self.file_object,
                "Successfully saved train and test data as train.csv and test.csv respectively.",
            )
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in split_dataset method of DataFetcher class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Dataset split unsuccessful. Exited split_dataset method of DataFetcher class",
            )
            raise Exception()
