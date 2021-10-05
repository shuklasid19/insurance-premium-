import json

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from config import Config
from evaluate_model import ModelScorer
from logger import AppLogger


class ModelTester:
    """
    This class shall be used for testing the best model.
    """

    def __init__(self, file_object):
        self.file_object = file_object
        self.logger_object = AppLogger()
        self.X_test = pd.read_csv(
            str(Config.FEATURES_PATH / "test_features_preprocessed.csv")
        )
        self.y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

    def load_best_model(self):
        """
        This method is used for loading the best model from joblib file.
        :return:
        """
        self.logger_object.log(
            self.file_object, "Entered load_best_model method of ModelTester class."
        )
        try:
            with open(str(Config.MODELS_PATH / "best_model.joblib"), "rb") as outfile:
                model = joblib.load(outfile)
                r_squared, rmse = ModelScorer(
                    file_object=self.file_object
                ).get_model_scores(model=model, X_test=self.X_test, y_test=self.y_test)
            self.logger_object.log(self.file_object, "Successfully loaded best model.")
            print(f"rsquared:{r_squared}, rmse:{rmse}")
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in load_best_model method of ModelTester class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Loading final model unsuccessful. Exited load_best_model method of ModelTester class",
            )
            raise Exception()
