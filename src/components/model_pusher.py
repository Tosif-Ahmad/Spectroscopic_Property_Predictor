"""Push an accepted absorption-wavelength regression model to S3."""
import sys

from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.proj1_estimator = Proj1Estimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path,
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """Upload only a model accepted by evaluation; retain its local copy."""
        try:
            if not self.model_evaluation_artifact.is_model_accepted:
                raise ValueError("Model evaluation rejected the new model; S3 upload cancelled.")

            if (self.model_evaluation_artifact.s3_model_path
                    != self.model_pusher_config.s3_model_key_path):
                raise ValueError("Evaluation and pusher S3 model keys do not match.")

            logging.info("Uploading accepted model to S3")
            self.proj1_estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_path,
                remove=False,
            )
            artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,
            )
            logging.info("Model pusher artifact: %s", artifact)
            return artifact
        except Exception as e:
            raise MyException(e, sys) from e
