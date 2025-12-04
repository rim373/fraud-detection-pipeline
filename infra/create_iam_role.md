# IAM Role Setup (Learner Lab)

We reuse the default **LabRole** pre-created by AWS Academy.

## Role used
- Name: `LabRole`
- Policies (already attached):
  - AmazonS3FullAccess
  - AmazonSageMakerFullAccess
  - CloudWatchFullAccess
  - AmazonKinesisFullAccess

## Usage
Select `LabRole` for:
- SageMaker Notebook Instances
- Lambda Functions
- Kinesis Firehose (if used)
