
import os
from datetime import date


# ============================================================
# MongoDB Configuration
# ============================================================

DATABASE_NAME = "Proj1"
COLLECTION_NAME = "Proj1-Data"
MONGODB_URL_KEY = "MONGODB_URL"


# ============================================================
# General Pipeline Configuration
# ============================================================

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME: str = "model.pkl"
PREPROCSSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

CURRENT_YEAR = date.today().year

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


# ============================================================
# Dataset Columns
# ============================================================

TARGET_COLUMN: str = "Absorption max (nm)"

MOLECULE_SMILES_COLUMN: str = "Chromophore"
SOLVENT_SMILES_COLUMN: str = "Solvent"

DROP_COLUMNS = ["_id"]


# ============================================================
# AWS Configuration
# ============================================================

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"

REGION_NAME = "eu-north-1"


# ============================================================
# Data Ingestion Constants
# ============================================================

DATA_INGESTION_COLLECTION_NAME: str = COLLECTION_NAME

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"

DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

# ============================================================
# Data Validation Constants
# ============================================================

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"


# ============================================================
# Data Transformation Constants
# ============================================================

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


# ============================================================
# Molecular Feature Engineering
# ============================================================

# Morgan fingerprints for both chromophore and solvent
MORGAN_RADIUS: int = 2
MORGAN_N_BITS: int = 512

# MACCS fingerprints use 167 bits per molecule.
MACCS_N_BITS: int = 167

# Total features: 2 * (512 + 167) = 1358


# ============================================================
# Model Trainer Constants
# ============================================================

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

# Random Forest Regressor hyperparameters
MODEL_TRAINER_N_ESTIMATORS: int = 200
MODEL_TRAINER_MAX_DEPTH: int | None = None
MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = 2
MODEL_TRAINER_MIN_SAMPLES_LEAF: int = 1
MODEL_TRAINER_MAX_FEATURES: float = 1.0
MODEL_TRAINER_RANDOM_STATE: int = 42
MODEL_TRAINER_N_JOBS: int = -1


# ============================================================
# Model Evaluation Constants
# ============================================================

# Initial acceptance thresholds (adjust after validation).
# Model Evaluation Constants (TESTING ONLY)

MODEL_EVALUATION_EXPECTED_R2_SCORE: float = -1.0
MODEL_EVALUATION_MAX_RMSE: float = 10000.0
MODEL_EVALUATION_MAX_MAE: float = 200.0


# ============================================================
# Model Registry / AWS S3g
# ============================================================

MODEL_BUCKET_NAME: str = "my-spectroscopy-mlopsproj"
MODEL_PUSHER_S3_KEY: str = "model-registry"
MODEL_S3_MODEL_KEY: str = f"{MODEL_PUSHER_S3_KEY}/{MODEL_FILE_NAME}"


# ============================================================
# Application Configuration
# ============================================================

APP_HOST: str = "0.0.0.0"
APP_PORT: int = 5000
