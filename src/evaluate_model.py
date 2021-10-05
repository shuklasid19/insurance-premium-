from sklearn.metrics import mean_squared_error, r2_score

from logger import AppLogger


class ModelScorer:
    """
    This class shall be used for getting scores of the trained model.
    """

    def __init__(self, file_object):
        self.file_object = file_object
        self.logger_object = AppLogger()

    def get_model_scores(self, model, X_test, y_test):
        """
        This method is used for evaluating the trained model.
        :param model: trained model to be evaluated
        :param X_test: features
        :param y_test: labels
        :return: r squared score and root mean squared error
        """
        self.logger_object.log(
            self.file_object, "Entered get_model_scores method of ModelScorer class"
        )
        try:
            y_pred = model.predict(X_test)
            self.logger_object.log(self.file_object, "Successfully predicted X_test.")
            r_squared = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            self.logger_object.log(
                self.file_object, "Successfully calculated model scores."
            )
            return r_squared, rmse
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in get_model_scores method of ModelScorer class. Exception message: {e}",
            )
            self.logger_object.log(
                self.file_object,
                "Model scoring unsuccessful. Exited get_model_scores method of ModelScorer class",
            )
            raise Exception()
