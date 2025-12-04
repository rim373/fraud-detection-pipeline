import boto3
import json
import pandas as pd
from io import StringIO
import logging
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
BUCKET = 'financial-fraud-project'
RAW_PREFIX = 'raw/'
PROCESSED_PREFIX = 'processed/'

def run_data_merge_and_split():  # ✅ CHANGED FROM lambda_handler
    """
    Merge raw JSON files and split into train/test sets
    
    Input: raw/*.json files
    Output: processed/train.csv, processed/test.csv
    """
    logger.info("🚀 Preprocessing started...")

    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=RAW_PREFIX)
    if 'Contents' not in response:
        logger.warning("⚠️ No raw files found in bucket.")
        return {"status": "no_raw_files"}

    all_records = []
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('.json'):
            data = s3.get_object(Bucket=BUCKET, Key=key)
            content = data['Body'].read().decode('utf-8')
            record = json.loads(content)
            if isinstance(record, list):
                all_records.extend(record)
            else:
                all_records.append(record)

    if not all_records:
        logger.warning("⚠️ No data in raw files.")
        return {"status": "no_data"}

    # Create DataFrame without filling missing values
    df = pd.DataFrame(all_records).drop_duplicates()

    if 'isFraud' not in df.columns:
        logger.error("❌ Missing 'isFraud' column.")
        return {"status": "error", "message": "Missing 'isFraud'"}

    # Simple 70/30 split without sklearn
    random.seed(42)
    msk = [random.random() < 0.7 for _ in range(len(df))]
    train_df = df[msk]
    test_df = df[~pd.Series(msk)]

    # Keep isFraud column in test set (do not drop)
    
    train_buf, test_buf = StringIO(), StringIO()
    train_df.to_csv(train_buf, index=False)
    test_df.to_csv(test_buf, index=False)

    s3.put_object(Bucket=BUCKET, Key=f"{PROCESSED_PREFIX}train.csv", Body=train_buf.getvalue())
    s3.put_object(Bucket=BUCKET, Key=f"{PROCESSED_PREFIX}test.csv", Body=test_buf.getvalue())

    logger.info(f"✅ Data split completed - Train: {len(train_df)}, Test: {len(test_df)}")

    return {
        "status": "success",
        "train_size": len(train_df),
        "test_size": len(test_df),
        "columns": list(df.columns)
    }