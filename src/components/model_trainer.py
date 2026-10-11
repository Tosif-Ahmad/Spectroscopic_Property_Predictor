"""Train and evaluate a Random Forest model for absorption wavelength prediction."""

import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact,
)
from src.entity.config_entity import ModelTrainerConfig
from src.entity.estimator import MyModel
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object


class ModelTrainer:
    """Train a regressor on chemical fingerprints and save its prediction wrapper."""

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetricArtifact:
        """Calculate R² and wavelength errors (RMSE, MAE in nm)."""
        return RegressionMetricArtifact(
            r2_score=float(r2_score(y_true, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            mae=float(mean_absolute_error(y_true, y_pred)),
        )

    def get_model_object_and_report(
        self, train: np.ndarray, test: np.ndarray
    ) -> tuple[RandomForestRegressor, RegressionMetricArtifact, RegressionMetricArtifact]:
        """Fit the regressor and report training and held-out test metrics."""
        try:
            if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
                raise ValueError("Training and test arrays must be 2D with matching columns")
            if train.shape[0] < 2 or test.shape[0] < 2 or train.shape[1] < 2:
                raise ValueError("Insufficient samples or features for regression evaluation")

            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            if not (np.isfinite(train).all() and np.isfinite(test).all()):
                raise ValueError("Transformed arrays contain non-finite values")

            config = self.model_trainer_config
            model = RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                max_features=config.max_features,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            )

            logging.info("Training RandomForestRegressor on %d samples", len(x_train))
            model.fit(x_train, y_train)

            train_metrics = self.calculate_metrics(y_train, model.predict(x_train))
            test_metrics = self.calculate_metrics(y_test, model.predict(x_test))
            logging.info("Train regression metrics: %s", train_metrics)
            logging.info("Test regression metrics: %s", test_metrics)
            return model, train_metrics, test_metrics

        except Exception as exc:
            raise MyException(exc, sys) from exc

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Load fingerprint arrays, train a regressor, and save the inference-ready model."""
        try:
            logging.info("Starting model trainer component")
            artifact = self.data_transformation_artifact
            train_arr = load_numpy_array_data(file_path=artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=artifact.transformed_test_file_path)

            trained_model, train_metrics, test_metrics = self.get_model_object_and_report(
                train=train_arr, test=test_arr
            )

            preprocessing_obj = load_object(file_path=artifact.transformed_object_file_path)
            inference_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=trained_model,
            )
            save_object(self.model_trainer_config.trained_model_file_path, inference_model)

            result = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
            )
            logging.info("Model trainer completed: %s", result)
            return result

        except Exception as exc:
            raise MyException(exc, sys) from exc
