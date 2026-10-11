"""Feature transformation for chromophore/solvent absorption regression."""
import sys
import numpy as np
import pandas as pd
from src.utils.chemical_features import ChemicalFingerprintTransformer
from src.constants import TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data

class DataTransformation:
    class DataTransformation:
        """
        Transform validated chemical datasets into machine-learning-ready features.

        Responsibilities:
            - Load cleaned training and testing datasets.
            - Generate Morgan and MACCS fingerprints for chromophores and solvents.
            - Separate input features from the absorption wavelength target.
            - Save transformed datasets as NumPy arrays.
            - Save the reusable fingerprint transformer for future predictions.
        """
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
