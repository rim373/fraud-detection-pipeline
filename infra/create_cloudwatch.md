# CloudWatch Log Group Setup Guide

## Purpose
Collects logs from Lambda functions, SageMaker jobs, and EC2 instances in the fraud detection pipeline.

## Configuration Steps
1. Open AWS Console → CloudWatch → Logs → Create log group
2. Name: `/fraud-detection/logs`
3. Retention: 7 days

## Result
✅ Log Group ARN: `arn:aws:logs:us-east-1:<ACCOUNT_ID>:log-group:/fraud-detection/logs`
