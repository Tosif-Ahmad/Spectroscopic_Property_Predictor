from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from src.constants import (
    MOLECULE_SMILES_COLUMN,
    SOLVENT_SMILES_COLUMN,
    MORGAN_RADIUS,
    MORGAN_N_BITS,
)

class ChemicalFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Reproducibly convert two SMILES columns to Morgan + MACCS bit vectors."""
    

    def __init__(
        self,
        molecule_column=MOLECULE_SMILES_COLUMN,
        solvent_column=SOLVENT_SMILES_COLUMN,
        radius=MORGAN_RADIUS,
        n_bits=MORGAN_N_BITS,
    ):
        """
            Initialize the chemical fingerprint transformer.

            Args:
                molecule_column (str): Column containing chromophore SMILES.
                solvent_column (str): Column containing solvent SMILES.
                radius (int): Radius used for Morgan fingerprint generation.
                n_bits (int): Number of bits in each Morgan fingerprint.
"""
        self.molecule_column = molecule_column
        self.solvent_column = solvent_column
        self.radius = radius
        self.n_bits = n_bits

    def fit(self, X, y=None):
        """
        Validate the input columns without learning any parameters.

        Args:
            X (pd.DataFrame): Input dataset containing SMILES columns.
            y (optional): Target values, unused.

        Returns:
            ChemicalFingerprintTransformer: The transformer instance.
        """
        self._check_columns(X)
        return self

    def _check_columns(self, X):
        """
        Check that the input is a DataFrame with the required SMILES columns.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
            ValueError: If any required SMILES column is missing.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame with SMILES columns")
        missing = [c for c in (self.molecule_column, self.solvent_column) if c not in X.columns]
        if missing:
            raise ValueError(f"Missing SMILES columns: {missing}")

    def _encode_smiles(self, smiles, generator, row_index, column):
        """
        Convert a single SMILES string into Morgan and MACCS fingerprints.

        Args:
            smiles (str): Molecular SMILES representation.
            generator: RDKit Morgan fingerprint generator.
            row_index: Index of the molecule in the input DataFrame.
            column (str): Name of the SMILES column.

        Returns:
            np.ndarray: Concatenated Morgan and MACCS fingerprint vector.

        Raises:
            ValueError: If the SMILES string is missing or invalid.
        """
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
        """
        Transform a DataFrame of SMILES strings into concatenated Morgan and MACCS fingerprints.

        Args:
            X (pd.DataFrame): Input DataFrame containing SMILES columns for molecules and solvents.

        Returns:
            np.ndarray: Array of concatenated fingerprints for each row in the input DataFrame.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
            ValueError: If any required SMILES column is missing or contains invalid SMILES strings.
        """
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
