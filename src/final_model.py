import json

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from config import Config
from evaluate_model import ModelScorer
from logger import AppLogger
from train_model import ModelTrainer


class ModelDumper:
    """
    This class shall be used for dumping best model as joblib file.
    """

    def __init__(self, file_object):
        self.file_object = file_object
        self.logger_object = AppLogger()
        self.X_train = pd.read_csv(
            str(Config.FEATURES_PATH / "train_features_preprocessed.csv")
        )
        self.y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

    def dump_best_model(self):
        """
        This method is used for dumping the best model as joblib file.
        :return:
        """
        self.logger_object.log(
            self.file_object, "Entered dump_best_model method of ModelDumper class."
        )
        try:
            with open(str(Config.TUNED_HYPERPARAMS_FILE_PATH), "r") as outfile:
                params = json.load(outfile)
                self.logger_object.log(
                    self.file_object,
                    "Successfully loaded parameters from best_params.json .",
                )
                model = LGBMRegressor(**params[0])
                model = ModelTrainer(file_object=self.file_object).get_trained_model(
                    model=model, X_train=self.X_train, y_train=self.y_train
                )
                self.logger_object.log(
                    self.file_object, "Successfully trained final model."
                )

            with open(str(Config.MODELS_PATH / "best_model.joblib"), "wb") as outfile:
                joblib.dump(model, outfile)
            self.logger_object.log(
                self.file_object,
                "Successfully dumped final model as best_model.joblib .",
            )
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in dump_best_model method of ModelDumper class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Dumping final model unsuccessful. Exited dump_best_model method of ModelDumper class",
            )
            raise Exception()
