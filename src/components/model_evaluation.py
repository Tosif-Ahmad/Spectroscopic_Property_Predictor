"""Evaluate a new absorption-wavelength regressor against the deployed S3 model."""
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.constants import MOLECULE_SMILES_COLUMN, SOLVENT_SMILES_COLUMN, TARGET_COLUMN
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object


@dataclass
class EvaluateModelResponse:
    """Metrics and deployment decision from a common evaluation dataset."""
    new_model_metrics: RegressionMetricArtifact
    production_model_metrics: Optional[RegressionMetricArtifact]
    is_model_accepted: bool


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_validation_artifact: Optional[DataValidationArtifact] = None,
    ):
        self.model_eval_config = model_eval_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.data_validation_artifact = data_validation_artifact

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """Return the S3 model wrapper, or None if no production model exists."""
        try:
            estimator = Proj1Estimator(
                bucket_name=self.model_eval_config.bucket_name,
                model_path=self.model_eval_config.s3_model_key_path,
            )
            if estimator.is_model_present(model_path=self.model_eval_config.s3_model_key_path):
                return estimator
            return None
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def _regression_metrics(actual, predicted) -> RegressionMetricArtifact:
        actual = np.asarray(actual, dtype=float).reshape(-1)
        predicted = np.asarray(predicted, dtype=float).reshape(-1)
        if actual.shape != predicted.shape or not np.isfinite(predicted).all():
            raise ValueError("Predictions must be finite and match the target shape.")
        return RegressionMetricArtifact(
            r2_score=float(r2_score(actual, predicted)),
            rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
            mae=float(mean_absolute_error(actual, predicted)),
        )

    def _load_evaluation_data(self):
        """Prefer the SMILES-validated held-out test file, if provided."""
        if self.data_validation_artifact is not None:
            if not self.data_validation_artifact.validation_status:
                raise ValueError("Data validation failed; model evaluation cannot continue.")
            path = self.data_validation_artifact.cleaned_test_file_path
        else:
            path = self.data_ingestion_artifact.test_file_path
            logging.warning(
                "No DataValidationArtifact supplied; using raw test data. "
                "Pass the validation artifact to use the cleaned test dataset."
            )

        df = pd.read_csv(path)
        required = [MOLECULE_SMILES_COLUMN, SOLVENT_SMILES_COLUMN, TARGET_COLUMN]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing evaluation columns: {missing}")
        df = df[required].copy()
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        df = df[df[TARGET_COLUMN] > 0]
        if df.empty:
            raise ValueError("No valid evaluation rows remain.")
        if self.data_validation_artifact is None:
            from rdkit import Chem
            valid = df[MOLECULE_SMILES_COLUMN].map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
            valid &= df[SOLVENT_SMILES_COLUMN].map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
            df = df.loc[valid]
            if df.empty:
                raise ValueError("No valid SMILES pairs remain in the test data.")
        return df[[MOLECULE_SMILES_COLUMN, SOLVENT_SMILES_COLUMN]], df[TARGET_COLUMN]

    def evaluate_model(self) -> EvaluateModelResponse:
        """Compare both models on identical held-out rows and apply acceptance gates."""
        try:
            x_test, y_test = self._load_evaluation_data()
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            new_metrics = self._regression_metrics(y_test, trained_model.predict(x_test))

            config = self.model_eval_config
            passes_thresholds = (
                np.isfinite(new_metrics.r2_score)
                and new_metrics.r2_score >= config.expected_r2_score
                and new_metrics.rmse <= config.max_rmse
                and new_metrics.mae <= config.max_mae
            )

            production_metrics = None
            production_model = self.get_best_model()
            if production_model is not None:
                production_metrics = self._regression_metrics(
                    y_test, production_model.predict(x_test)
                )
                # Require no degradation in RMSE and MAE relative to production.
                better_than_production = (
                    new_metrics.rmse < production_metrics.rmse
                    and new_metrics.mae <= production_metrics.mae
                )
            else:
                better_than_production = True

            accepted = bool(passes_thresholds and better_than_production)
            logging.info(
                "Model evaluation: new=%s, production=%s, accepted=%s",
                new_metrics, production_metrics, accepted,
            )
            return EvaluateModelResponse(
                new_model_metrics=new_metrics,
                production_model_metrics=production_metrics,
                is_model_accepted=accepted,
            )
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Create the artifact consumed by the model-pusher component."""
        try:
            result = self.evaluate_model()
            artifact = ModelEvaluationArtifact(
                is_model_accepted=result.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                new_model_metrics=result.new_model_metrics,
                production_model_metrics=result.production_model_metrics,
            )
            logging.info("Model evaluation artifact: %s", artifact)
            return artifact
        except Exception as e:
            raise MyException(e, sys) from e
