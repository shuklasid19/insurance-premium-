import json

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from config import Config
from evaluate_model import ModelScorer
from logger import AppLogger
from train_model import ModelTrainer


class ModelMetricsGenerator:
    def __init__(self, file_object):
        self.file_object = file_object
        self.logger_object = AppLogger()
        self.models_dict = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regression": DecisionTreeRegressor(),
            "SVR": SVR(),
            "Random Forest Regression": RandomForestRegressor(),
            "Extra Trees Regression": ExtraTreesRegressor(),
            "LGBM Regression": LGBMRegressor(),
            "XGB Regression": XGBRegressor(),
        }
        self.metrics = {"models": []}
        self.X_train = pd.read_csv(
            str(Config.FEATURES_PATH / "train_features_preprocessed.csv")
        )
        self.X_test = pd.read_csv(
            str(Config.FEATURES_PATH / "test_features_preprocessed.csv")
        )
        self.y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))
        self.y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

    def create_model_metrics(self):
        """
        This method generates a json file containing name and scores of each model.
        :return: None
        """
        self.logger_object.log(
            self.file_object,
            "Entered create_model_metrics method of ModelMetricsGenerator class.",
        )
        try:
            for model_name, model in self.models_dict.items():
                model = ModelTrainer(file_object=self.file_object).get_trained_model(
                    model=model, X_train=self.X_train, y_train=self.y_train
                )
                r_squared, rmse = ModelScorer(
                    file_object=self.file_object
                ).get_model_scores(model=model, X_test=self.X_test, y_test=self.y_test)

                self.metrics["models"].append(
                    {"model_name": model_name, "r_squared": r_squared, "rmse": rmse}
                )
            self.logger_object.log(
                self.file_object,
                "Successfully appended model name and model scores as a dictionary.",
            )

            with (open(str(Config.METRICS_FILE_PATH), "w")) as outfile:
                json.dump(self.metrics["models"], outfile, indent=1)
            self.logger_object.log(
                self.file_object,
                "Successfully dumped the metrics dictionaries in metrics.json .",
            )
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in create_model_metrics method of ModelMetricsGenerator class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Dumping model metrics unsuccessful. Exited create_model_metrics method of ModelMetricsGenerator class",
            )
            raise Exception()
