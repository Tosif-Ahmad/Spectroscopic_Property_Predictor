"""Feature transformation for chromophore/solvent absorption regression."""
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin

from src.constants import (
    TARGET_COLUMN,
    MOLECULE_SMILES_COLUMN,
    SOLVENT_SMILES_COLUMN,
    MORGAN_RADIUS,
    MORGAN_N_BITS,
)
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data


class ChemicalFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Reproducibly convert two SMILES columns to Morgan + MACCS bit vectors."""

    def __init__(
        self,
        molecule_column=MOLECULE_SMILES_COLUMN,
        solvent_column=SOLVENT_SMILES_COLUMN,
        radius=MORGAN_RADIUS,
        n_bits=MORGAN_N_BITS,
    ):
        self.molecule_column = molecule_column
        self.solvent_column = solvent_column
        self.radius = radius
        self.n_bits = n_bits

    def fit(self, X, y=None):
        self._check_columns(X)
        return self

    def _check_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame with SMILES columns")
        missing = [c for c in (self.molecule_column, self.solvent_column) if c not in X.columns]
        if missing:
            raise ValueError(f"Missing SMILES columns: {missing}")

    def _encode_smiles(self, smiles, generator, row_index, column):
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"Missing SMILES at row {row_index}, column {column}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {row_index}, column {column}: {smiles!r}")

        morgan = generator.GetFingerprint(mol)
        morgan_array = np.zeros((self.n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(morgan, morgan_array)

        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_array = np.zeros((167,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(maccs, maccs_array)
        return np.concatenate((morgan_array, maccs_array))

    def transform(self, X):
        self._check_columns(X)
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits
        )
        rows = []
        for row_index, row in X.iterrows():
            molecule_fp = self._encode_smiles(
                row[self.molecule_column], generator, row_index, self.molecule_column
            )
            solvent_fp = self._encode_smiles(
                row[self.solvent_column], generator, row_index, self.solvent_column
            )
            rows.append(np.concatenate((molecule_fp, solvent_fp)))

        return np.vstack(rows).astype(np.float64) if rows else np.empty(
            (0, 2 * (self.n_bits + 167)), dtype=np.float64
        )


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def get_data_transformer_object(self) -> ChemicalFingerprintTransformer:
        return ChemicalFingerprintTransformer()

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise ValueError(self.data_validation_artifact.message)

            train_df = self.read_data(self.data_validation_artifact.cleaned_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.cleaned_test_file_path)
            if train_df.empty or test_df.empty:
                raise ValueError("Cleaned training and test datasets must both contain rows")

            y_train = pd.to_numeric(train_df[TARGET_COLUMN], errors="raise").to_numpy(dtype=np.float64)
            y_test = pd.to_numeric(test_df[TARGET_COLUMN], errors="raise").to_numpy(dtype=np.float64)
            if not (np.isfinite(y_train).all() and np.isfinite(y_test).all()):
                raise ValueError("Targets contain non-finite values")
            if not ((y_train > 0).all() and (y_test > 0).all()):
                raise ValueError("Absorption wavelengths must be positive")

            preprocessor = self.get_data_transformer_object()
            X_train = preprocessor.fit_transform(train_df)
            X_test = preprocessor.transform(test_df)

            train_arr = np.column_stack((X_train, y_train))
            test_arr = np.column_stack((X_test, y_test))

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            logging.info(
                "Chemical fingerprint transformation complete: train=%s, test=%s",
                train_arr.shape, test_arr.shape,
            )
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise MyException(e, sys) from e
