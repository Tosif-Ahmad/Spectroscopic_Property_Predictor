import sys

import numpy as np
import pandas as pd

from src.exception import MyException
from src.logger import logging


class MyModel:
    """
    Combine chemical fingerprint preprocessing and a trained regression model.

    The preprocessing object converts chromophore and solvent SMILES
    into Morgan and MACCS fingerprints.

    The trained model predicts absorption wavelength in nanometers.
    """

    def __init__(self, preprocessing_object: object, trained_model_object: object):
        """
        Initialize the prediction wrapper.

        Args:
            preprocessing_object: Fitted ChemicalFingerprintTransformer.
            trained_model_object: Trained regression model.
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Predict absorption wavelength from chromophore and solvent SMILES.

        Args:
            dataframe (pd.DataFrame): Input containing Chromophore and Solvent.

        Returns:
            np.ndarray: Predicted absorption wavelengths in nanometers.
        """
        try:
            logging.info("Starting absorption wavelength prediction.")

            # Convert SMILES into Morgan + MACCS fingerprint features.
            transformed_features = self.preprocessing_object.transform(dataframe)

            # Predict absorption wavelength using the trained regressor.
            predictions = self.trained_model_object.predict(transformed_features)

            logging.info("Absorption wavelength prediction completed.")

            return predictions

        except Exception as e:
            logging.error("Prediction failed.", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"