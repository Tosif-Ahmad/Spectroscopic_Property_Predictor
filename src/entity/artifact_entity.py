from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str
    # NEW: Cleaned dataset paths
    cleaned_train_file_path: str
    cleaned_test_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str


@dataclass
class RegressionMetricArtifact:
    """
    Stores regression performance metrics.

    R² measures predictive performance.
    RMSE and MAE measure prediction errors in nanometers.
    """
    r2_score: float
    rmse: float
    mae: float


@dataclass
class ModelTrainerArtifact:
    """
    Stores the trained model location and its performance
    on training and testing datasets.
    """
    trained_model_file_path: str
    train_metric_artifact: RegressionMetricArtifact
    test_metric_artifact: RegressionMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    """
    Stores the outcome of comparing a newly trained model
    against the existing production model.
    """

    is_model_accepted: bool

    trained_model_path: str
    s3_model_path: str

    new_model_metrics: RegressionMetricArtifact
    production_model_metrics: RegressionMetricArtifact | None = None


@dataclass
class ModelPusherArtifact:
    """Stores the S3 location of the uploaded regression model."""

    bucket_name: str
    s3_model_path: str