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

MODEL_FILE_NAME = "model.pkl"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

CURRENT_YEAR = date.today().year

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


# ============================================================
# Dataset Columns
# ============================================================

# Target variable (regression)
TARGET_COLUMN = "Absorption max (nm)"

# Molecular structure columns
MOLECULE_SMILES_COLUMN = "Chromophore"
SOLVENT_SMILES_COLUMN = "Solvent"

# MongoDB identifier (removed during ingestion)
DROP_COLUMNS = ["_id"]


# ============================================================
# AWS Configuration
# ============================================================

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"

REGION_NAME = "us-east-1"


# ============================================================
# Data Ingestion Constants
# ============================================================

DATA_INGESTION_COLLECTION_NAME: str = "Proj1-Data"

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

# Morgan fingerprint configuration
MORGAN_RADIUS: int = 2
MORGAN_N_BITS: int = 512

# RDKit descriptor configuration
USE_MOLECULE_DESCRIPTORS: bool = True
USE_SOLVENT_DESCRIPTORS: bool = True

# Solvent fingerprints (optional)
USE_SOLVENT_FINGERPRINTS: bool = False


# ============================================================
# Model Trainer Constants
# ============================================================

MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

# Regression model performance threshold (R²)
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join(
    "config", "model.yaml"
)

# Random Forest Regressor hyperparameters
MODEL_TRAINER_N_ESTIMATORS: int = 200

MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = 7

MODEL_TRAINER_MIN_SAMPLES_LEAF: int = 6

# Preserved original constant names for compatibility
MIN_SAMPLES_SPLIT_MAX_DEPTH: int = 10

MIN_SAMPLES_SPLIT_CRITERION: str = "squared_error"

MIN_SAMPLES_SPLIT_RANDOM_STATE: int = 101


# ============================================================
# Model Evaluation Constants
# ============================================================

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02


# ============================================================
# Model Registry / AWS S3
# ============================================================

MODEL_BUCKET_NAME = "my-model-mlopsproj"

MODEL_PUSHER_S3_KEY = "model-registry"


# ============================================================
# Application Configuration
# ============================================================

APP_HOST = "0.0.0.0"

APP_PORT = 5000