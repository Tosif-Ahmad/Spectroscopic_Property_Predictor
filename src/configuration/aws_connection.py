import boto3


class S3Client:
    """Create an S3 client using AWS's default credential chain."""

    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            region_name="eu-north-1"
        )

        self.s3_resource = boto3.resource(
            "s3",
            region_name="eu-north-1"
        )