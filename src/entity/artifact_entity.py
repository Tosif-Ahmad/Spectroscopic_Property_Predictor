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
# @dataclass
# class DataValidationArtifact:
#     validation_status:bool
#     message: str
#     validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str

from dataclasses import dataclass


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

# @dataclass
# class ClassificationMetricArtifact:
#     f1_score:float
#     precision_score:float
#     recall_score:float

# @dataclass
# class ModelTrainerArtifact:
#     trained_model_file_path:str 
#     metric_artifact:ClassificationMetricArtifact

# @dataclass
# class ModelEvaluationArtifact:
#     is_model_accepted:bool
#     changed_accuracy:float
#     s3_model_path:str 
#     trained_model_path:str

# @dataclass
# class ModelPusherArtifact:
#     bucket_name:str
#     s3_model_path:str