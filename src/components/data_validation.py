"""Validate and clean molecule/solvent SMILES regression data.

The ingestion train/test split is preserved; invalid rows are removed separately.
The transformation component must consume the cleaned paths, not ingestion paths.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame
from rdkit import Chem, RDLogger

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        self.smiles_columns = self._schema_config["smiles_columns"]
        self.target_column = self._schema_config["target_column"][0]

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        return pd.read_csv(file_path)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """Check expected columns by name (not just count)."""
        expected = {next(iter(item)) for item in self._schema_config["columns"]}
        dropped = self._schema_config.get("drop_columns", [])
        if isinstance(dropped, str):
            dropped = [dropped]
        expected -= set(dropped)
        return set(dataframe.columns) == expected

    def is_column_exist(self, df: DataFrame) -> bool:
        required = set(self.smiles_columns + [self.target_column])
        missing = sorted(required - set(df.columns))
        if missing:
            logging.error(f"Missing required columns: {missing}")
        return not missing

    def find_invalid_smiles(self, df: DataFrame) -> DataFrame:
        """Record every invalid SMILES, including both columns in one row."""
        records = []
        # Avoid noisy RDKit parse messages; reasons are recorded in the CSV.
        RDLogger.DisableLog("rdApp.error")
        try:
            for column in self.smiles_columns:
                for index, value in df[column].items():
                    reason = None
                    if not isinstance(value, str) or not value.strip():
                        reason = "missing_or_blank"
                    else:
                        try:
                            if Chem.MolFromSmiles(value.strip()) is None:
                                reason = "invalid_smiles"
                        except Exception:
                            reason = "invalid_smiles"
                    if reason:
                        records.append({
                            "row_index": int(index),
                            "csv_line_number": int(index) + 2,
                            "column": column,
                            "smiles": None if pd.isna(value) else str(value),
                            "reason": reason,
                        })
        finally:
            RDLogger.EnableLog("rdApp.error")
        return pd.DataFrame(records, columns=[
            "row_index", "csv_line_number", "column", "smiles", "reason"
        ])

    def find_invalid_targets(self, df: DataFrame) -> DataFrame:
        """Missing, non-numeric, infinite or nonpositive wavelengths."""
        numeric = pd.to_numeric(df[self.target_column], errors="coerce")
        invalid_mask = numeric.isna() | ~np.isfinite(numeric) | (numeric <= 0)
        return pd.DataFrame({
            "row_index": df.index[invalid_mask].astype(int),
            "csv_line_number": df.index[invalid_mask].astype(int) + 2,
            "value": df.loc[invalid_mask, self.target_column].to_numpy(),
            "reason": "missing_nonnumeric_nonfinite_or_nonpositive",
        })

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting SMILES validation and cleaning")
            report_path = self.data_validation_config.validation_report_file_path
            report_dir = os.path.dirname(report_path) or "."
            os.makedirs(report_dir, exist_ok=True)
            cleaned_dir = os.path.join(report_dir, "cleaned")
            os.makedirs(cleaned_dir, exist_ok=True)

            sources = {
                "train": self.data_ingestion_artifact.trained_file_path,
                "test": self.data_ingestion_artifact.test_file_path,
            }
            errors = []
            report = {}
            cleaned_paths = {}

            for split, source_path in sources.items():
                df = self.read_data(source_path)
                if not self.validate_number_of_columns(df):
                    errors.append(f"{split}: columns do not exactly match schema")
                if not self.is_column_exist(df):
                    errors.append(f"{split}: missing required columns")
                if errors:
                    # Do not attempt chemistry validation if schema is incompatible.
                    break

                invalid_smiles = self.find_invalid_smiles(df)
                invalid_targets = self.find_invalid_targets(df)
                invalid_smiles.to_csv(
                    os.path.join(report_dir, f"invalid_{split}_smiles.csv"), index=False
                )
                invalid_targets.to_csv(
                    os.path.join(report_dir, f"invalid_{split}_targets.csv"), index=False
                )

                remove_indices = set(invalid_smiles["row_index"].tolist())
                remove_indices.update(invalid_targets["row_index"].tolist())
                cleaned_df = df.drop(index=list(remove_indices)).copy()
                cleaned_df[self.target_column] = pd.to_numeric(
                    cleaned_df[self.target_column], errors="raise"
                )

                if cleaned_df.empty:
                    errors.append(f"{split}: no valid rows remain after cleaning")

                cleaned_path = os.path.join(cleaned_dir, f"cleaned_{split}.csv")
                cleaned_df.to_csv(cleaned_path, index=False)
                cleaned_paths[split] = cleaned_path
                report[split] = {
                    "original_rows": len(df),
                    "invalid_smiles_entries": len(invalid_smiles),
                    "invalid_target_rows": len(invalid_targets),
                    "removed_unique_rows": len(remove_indices),
                    "cleaned_rows": len(cleaned_df),
                    "cleaned_file_path": cleaned_path,
                }
                logging.info(f"{split}: removed {len(remove_indices)} rows; "
                             f"kept {len(cleaned_df)} rows")

            status = not errors
            message = "; ".join(errors) if errors else "Validation and cleaning completed"
            with open(report_path, "w", encoding="utf-8") as file:
                json.dump({"validation_status": status, "message": message,
                           "splits": report}, file, indent=4)

            # Requires the two optional paths to be added to DataValidationArtifact.
            return DataValidationArtifact(
                validation_status=status,
                message=message,
                validation_report_file_path=report_path,
                cleaned_train_file_path=cleaned_paths.get("train", ""),
                cleaned_test_file_path=cleaned_paths.get("test", ""),
            )
        except Exception as e:
            raise MyException(e, sys) from e
