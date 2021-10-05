import pandas as pd

from config import Config
from create_dataset import DatasetFetcher
from create_features import FeatureExtractor
from create_model_metrics import ModelMetricsGenerator
from final_model import ModelDumper
from logger import AppLogger
from preprocess_features import FeaturePreprocessor
from test_final_model import ModelTester
from tune_hyperparams import HyperparametersTuner


class TrainingPipeline:
    """
    This class uses  other scripts to create a complete model training pipeline.
    """

    def __init__(self):
        self.file_object = open(str(Config.LOGS_PATH / "train.log"), "a+")
        self.logger_object = AppLogger()

    def train_pipeline(self):
        """
        This method is used to run model training pipeline.
        :return: None
        """
        self.logger_object.log(
            self.file_object, "Entered train_pipeline method of TrainingPipeline class."
        )
        try:
            self.logger_object.log(
                self.file_object, "Successfully started model training pipeline."
            )
            dataset_fetcher_object = DatasetFetcher(file_object=self.file_object)
            dataset_fetcher_object.fetch_dataset()
            dataset_fetcher_object.split_dataset(test_size=0.2)

            feature_extractor_object = FeatureExtractor(file_object=self.file_object)
            feature_extractor_object.extract_features()
            feature_extractor_object.extract_labels()

            self.train_features = pd.read_csv(
                str(Config.FEATURES_PATH / "train_features.csv")
            )
            self.test_features = pd.read_csv(
                str(Config.FEATURES_PATH / "test_features.csv")
            )

            feature_preprocessor_object = FeaturePreprocessor(
                file_object=self.file_object
            )

            self.num_cols = feature_preprocessor_object.get_num_features(
                self.train_features
            )
            self.cat_cols = feature_preprocessor_object.get_cat_features(
                self.train_features
            )

            feature_preprocessor_object.preprocess_num_features(
                train_df=self.train_features,
                test_df=self.test_features,
                num_cols=self.num_cols,
            )
            feature_preprocessor_object.preprocess_cat_features(
                train_df=self.train_features,
                test_df=self.test_features,
                cat_cols=self.cat_cols,
            )

            feature_preprocessor_object.drop_features(
                features_df=self.train_features, drop_cols=self.cat_cols
            )
            feature_preprocessor_object.drop_features(
                features_df=self.test_features, drop_cols=self.cat_cols
            )

            feature_preprocessor_object.save_preprocessed_features(
                features_df=self.train_features,
                outfile_name="train_features_preprocessed",
            )
            feature_preprocessor_object.save_preprocessed_features(
                features_df=self.test_features,
                outfile_name="test_features_preprocessed",
            )

            model_metrics_generator_object = ModelMetricsGenerator(
                file_object=self.file_object
            )
            model_metrics_generator_object.create_model_metrics()

            hyperparameters_tuner_object = HyperparametersTuner(
                file_object=self.file_object
            )
            hyperparameters_tuner_object.find_best_params(n_trials=300)

            model_dumper_object = ModelDumper(file_object=self.file_object)
            model_dumper_object.dump_best_model()

            model_tester_object = ModelTester(file_object=self.file_object)
            model_tester_object.load_best_model()
            self.logger_object.log(
                self.file_object, "Successfully executed model training pipeline."
            )
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in train_pipeline method of TrainingPipeline class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Model training pipeline unsuccessful. Exited train_pipeline method of TrainingPipeline class",
            )
            raise Exception()


if __name__ == "__main__":
    training_pipeline_object = TrainingPipeline()
    training_pipeline_object.train_pipeline()
