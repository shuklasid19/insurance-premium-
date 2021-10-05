import json
from functools import partial

import optuna
import pandas as pd
from lightgbm import LGBMRegressor

from config import Config
from evaluate_model import ModelScorer
from logger import AppLogger
from train_model import ModelTrainer


class HyperparametersTuner:
    """
    This class shall be used for tuning the hyperparameters.
    """

    def __init__(self, file_object):
        self.file_object = file_object
        self.logger_object = AppLogger()
        self.tuned_hyperparams = {"LGBM Regression": []}
        self.X_train = pd.read_csv(
            str(Config.FEATURES_PATH / "train_features_preprocessed.csv")
        )
        self.X_test = pd.read_csv(
            str(Config.FEATURES_PATH / "test_features_preprocessed.csv")
        )
        self.y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))
        self.y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

    def optimize(self, trial):
        """
        This method is used by finding best parameters for maximizing r squared score.
        :return: r squared score
        """
        num_leaves = trial.suggest_int("num_leaves", 6, 50)
        min_child_samples = trial.suggest_int("min_child_samples", 100, 500)
        min_child_weight = trial.suggest_uniform("min_child_weight", 1, 7)
        subsample = trial.suggest_uniform("subsample", 0.6, 1)
        colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.6, 1)
        reg_alpha = trial.suggest_uniform("reg_alpha", 0.1, 100)
        reg_lambda = trial.suggest_uniform("reg_lambda", 0.1, 100)

        model = LGBMRegressor(
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

        model = ModelTrainer(file_object=self.file_object).get_trained_model(
            model, self.X_train, self.y_train
        )
        r_squared, rmse = ModelScorer(file_object=self.file_object).get_model_scores(
            model, self.X_test, self.y_test
        )

        return r_squared

    def find_best_params(self, n_trials=120):
        """
        This method is used for running the optimize function with given trials and save best parameters into json file.
        :param n_trials: no. of trials
        :return: None
        """
        self.logger_object.log(
            self.file_object,
            "Entered find_best_params method of HyperparametersTuner class.",
        )
        try:
            optimization_function = partial(self.optimize)
            study = optuna.create_study(direction="maximize")
            study.optimize(optimization_function, n_trials=n_trials)
            self.logger_object.log(
                self.file_object, f"Successfully ran {n_trials} optuna study trials."
            )

            self.tuned_hyperparams["LGBM Regression"].append(study.best_params)
            self.logger_object.log(
                self.file_object,
                "Successfully appended best model parameters as a dictionary.",
            )

            with (open(str(Config.TUNED_HYPERPARAMS_FILE_PATH), "w")) as outfile:
                json.dump(self.tuned_hyperparams["LGBM Regression"], outfile, indent=1)
            self.logger_object.log(
                self.file_object,
                "Successfully dumped the best parameters in best_params.json .",
            )
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in find_best_params method of HyperparametersTuner class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Dumping best parameters unsuccessful. Exited find_best_params method of HyperparametersTuner class",
            )
            raise Exception()
