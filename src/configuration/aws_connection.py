import boto3
from src.constants import REGION_NAME


class S3Client:
    """Create an S3 connection using AWS's default credential chain."""

    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):

        if S3Client.s3_client is None:
            S3Client.s3_client = boto3.client(
                "s3",
                region_name=region_name
            )

        if S3Client.s3_resource is None:
            S3Client.s3_resource = boto3.resource(
                "s3",
                region_name=region_name
            )

        self.s3_client = S3Client.s3_client
        self.s3_resource = S3Client.s3_resource