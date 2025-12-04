# S3 Bucket Setup Guide

## Purpose
Stores raw, processed, and model data for the fraud detection pipeline.

## Configuration Steps
1. Open AWS Console → S3 → Create Bucket
2. Name: `fraud-data-bucket`
3. Region: `us-east-1`
4. Enable versioning
5. Keep "Block all public access" ON (safe for internal AWS use)
6. Create the following folders:
   - `raw/`
   - `processed/`
   - `models/`

## Result
✅ S3 Bucket ARN: `arn:aws:s3:::rim-fraud-data-bucket`
