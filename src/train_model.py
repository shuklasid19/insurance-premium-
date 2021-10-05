from config import Config
from logger import AppLogger


class ModelTrainer:
    """
    This class shall be used for training the model.
    """

    def __init__(self, file_object):
        self.file_object = file_object
        self.logger_object = AppLogger()
        Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

    def get_trained_model(self, model, X_train, y_train):
        """
        This method is used for training the model.
        :param model: model to be trained
        :param X_train: features
        :param y_train: labels
        :return: trained model
        """
        self.logger_object.log(
            self.file_object, "Entered get_trained_model method of ModelTrainer class"
        )
        try:
            model = model.fit(X_train, y_train.to_numpy().ravel())
            self.logger_object.log(self.file_object, "Successfully trained model.")
            return model
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in get_trained_model method of ModelTrainer class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Training model unsuccessful. Exited get_trained_model method of ModelTrainer class",
            )
            raise Exception()
