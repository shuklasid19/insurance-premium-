from pathlib import Path


class Config:
    """
    This class shall be used for getting the configuration for the scripts.
    """

    RANDOM_SEED = 42
    ASSETS_PATH = Path("./assets")
    DATASET_URL = "https://www.kaggle.com/noordeen/insurance-premium-prediction"
    ORIGINAL_DATASET_FILE_PATH = (
        ASSETS_PATH
        / "original_dataset"
        / "insurance-premium-prediction"
        / "insurance.csv"
    )
    DATASET_PATH = ASSETS_PATH / "data"
    FEATURES_PATH = ASSETS_PATH / "features"
    MODELS_PATH = ASSETS_PATH / "models"
    METRICS_FILE_PATH = ASSETS_PATH / "metrics.json"
    LOGS_PATH = ASSETS_PATH / "logs"
    TUNED_HYPERPARAMS_FILE_PATH = ASSETS_PATH / "best_params.json"
    NOTEBOOKS_PATH = ASSETS_PATH / "notebooks"
