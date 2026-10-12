"""Inference pipeline for molecular absorption maximum wavelength (nm)."""

import sys

import numpy as np
from pandas import DataFrame
from rdkit import Chem

from src.constants import MOLECULE_SMILES_COLUMN, SOLVENT_SMILES_COLUMN
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging


class MoleculeData:
    """Prepare one chromophore/solvent SMILES pair for prediction."""

    def __init__(self, chromophore: str, solvent: str):
        self.chromophore = self._validate_smiles(chromophore, "Chromophore")
        self.solvent = self._validate_smiles(solvent, "Solvent")

    @staticmethod
    def _validate_smiles(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} SMILES must be a non-empty string.")
        smiles = value.strip()
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"Invalid {label} SMILES: {smiles}")
        return smiles

    def get_molecule_data_as_dict(self) -> dict[str, list[str]]:
        """Use the exact feature names expected by the saved preprocessor."""
        return {
            MOLECULE_SMILES_COLUMN: [self.chromophore],
            SOLVENT_SMILES_COLUMN: [self.solvent],
        }

    def get_molecule_input_data_frame(self) -> DataFrame:
        return DataFrame(self.get_molecule_data_as_dict())


class AbsorptionWavelengthPredictor:
    """Predict absorption maxima with the deployed S3 regression model."""

    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig | None = None):
        # Keep VehiclePredictorConfig for compatibility with existing config_entity.py.
        self.prediction_pipeline_config = prediction_pipeline_config or VehiclePredictorConfig()
        self._estimator: Proj1Estimator | None = None

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        """Return one predicted absorption wavelength (nm) per input row."""
        try:
            required = [MOLECULE_SMILES_COLUMN, SOLVENT_SMILES_COLUMN]
            missing = [column for column in required if column not in dataframe.columns]
            if missing:
                raise ValueError(f"Missing SMILES columns: {missing}")
            if dataframe.empty:
                raise ValueError("At least one molecule/solvent pair is required.")

            # Validate all rows before calling the deployed model.
            features = dataframe.loc[:, required].copy()
            for column in required:
                features[column] = features[column].map(
                    lambda value: MoleculeData._validate_smiles(value, column)
                )

            if self._estimator is None:
                config = self.prediction_pipeline_config
                self._estimator = Proj1Estimator(
                    bucket_name=config.model_bucket_name,
                    model_path=config.model_file_path,
                )

            predictions = np.asarray(self._estimator.predict(features), dtype=float).reshape(-1)
            if len(predictions) != len(features) or not np.isfinite(predictions).all():
                raise ValueError("Model returned invalid absorption wavelength predictions.")
            logging.info("Predicted absorption wavelength for %d sample(s).", len(features))
            return predictions
        except Exception as e:
            raise MyException(e, sys) from e
