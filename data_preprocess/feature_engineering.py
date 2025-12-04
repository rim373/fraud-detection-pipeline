import boto3
import pandas as pd
import numpy as np
from io import StringIO
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
BUCKET = 'financial-fraud-project'
PROCESSED_PREFIX = 'processed/'
FEATURED_PREFIX = 'featured/'

def read_csv_from_s3(bucket, key):
    """Read CSV file from S3"""
    logger.info(f"Reading s3://{bucket}/{key}...")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'])

def save_csv_to_s3(df, bucket, key):
    """Save DataFrame to S3 as CSV"""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue()
    )
    logger.info(f"✅ Saved to s3://{bucket}/{key}")

def simple_label_encode(series):
    """Simple label encoding without sklearn"""
    unique_values = series.unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    return series.map(mapping), mapping

def preprocess_data(df, is_train=True):
    """
    Preprocess the fraud detection dataset
    """
    df = df.copy()
    
    # Store TransactionID for later if needed
    transaction_id = None
    if 'TransactionID' in df.columns:
        transaction_id = df['TransactionID'].copy()
        df = df.drop(columns=['TransactionID'])
    
    # Separate target if training data
    target = None
    if is_train and 'isFraud' in df.columns:
        target = df['isFraud'].copy()
        df = df.drop(columns=['isFraud'])
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numerical columns count: {len(numerical_cols)}")
    
    # Handle categorical variables - Simple Label Encoding
    label_mappings = {}
    for col in categorical_cols:
        # Handle missing values by filling with 'missing' before encoding
        df[col] = df[col].fillna('missing').astype(str)
        df[col], mapping = simple_label_encode(df[col])
        label_mappings[col] = mapping
    
    # Handle missing values in numerical columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            # Fill with median for numerical columns
            df[col] = df[col].fillna(df[col].median())
    
    # Fill any remaining missing values with 0
    df = df.fillna(0)
    
    logger.info(f"Processed shape: {df.shape}")
    logger.info(f"Remaining missing values: {df.isnull().sum().sum()}")
    
    return df, target, transaction_id, label_mappings

def run_feature_engineering():
    """
    Main function to run feature engineering pipeline
    """
    logger.info("⚙️ Feature Engineering started...")
    
    try:
        # ========================================
        # Load datasets from processed/ folder
        # ========================================
        logger.info("Loading data from S3...")
        train = read_csv_from_s3(BUCKET, f"{PROCESSED_PREFIX}train.csv")
        test = read_csv_from_s3(BUCKET, f"{PROCESSED_PREFIX}test.csv")
        
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")
        logger.info(f"Train columns: {train.columns.tolist()}")
        
        # ========================================
        # Check fraud distribution
        # ========================================
        if 'isFraud' in train.columns:
            fraud_counts = train['isFraud'].value_counts()
            logger.info(f"Fraud distribution in train: {fraud_counts.to_dict()}")
        
        # ========================================
        # Check data types and missing values
        # ========================================
        logger.info("Data types:")
        logger.info(str(train.dtypes.to_dict()))
        
        missing_train = train.isnull().sum()
        missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
        if len(missing_train) > 0:
            logger.info(f"Missing values in train: {missing_train.to_dict()}")
        
        missing_test = test.isnull().sum()
        missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
        if len(missing_test) > 0:
            logger.info(f"Missing values in test: {missing_test.to_dict()}")
        
        # ========================================
        # Preprocess training data
        # ========================================
        logger.info("="*50)
        logger.info("PREPROCESSING TRAINING DATA")
        logger.info("="*50)
        X_train, y_train, train_ids, train_mappings = preprocess_data(train, is_train=True)
        
        # ========================================
        # Preprocess test data
        # ========================================
        logger.info("="*50)
        logger.info("PREPROCESSING TEST DATA")
        logger.info("="*50)
        X_test, y_test, test_ids, test_mappings = preprocess_data(test, is_train=True)
        
        # ========================================
        # Ensure train and test have the same columns
        # ========================================
        logger.info(f"Train features: {X_train.shape[1]}")
        logger.info(f"Test features: {X_test.shape[1]}")
        
        # Get common columns
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        logger.info(f"Common features: {len(common_cols)}")
        
        # Align columns
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        
        logger.info(f"Final train shape: {X_train.shape}")
        logger.info(f"Final test shape: {X_test.shape}")
        
        # ========================================
        # Check for infinite values
        # ========================================
        logger.info("Checking for infinite values...")
        train_inf = np.isinf(X_train).sum().sum()
        test_inf = np.isinf(X_test).sum().sum()
        logger.info(f"Train infinite values: {train_inf}")
        logger.info(f"Test infinite values: {test_inf}")
        
        # Replace infinite values if any
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        
        logger.info("Data quality check passed!")
        
        # ========================================
        # Save preprocessed data to S3
        # ========================================
        logger.info("="*50)
        logger.info("SAVING PREPROCESSED DATA TO S3")
        logger.info("="*50)
        
        save_csv_to_s3(X_train, BUCKET, f'{FEATURED_PREFIX}X_train_preprocessed.csv')
        save_csv_to_s3(pd.DataFrame(y_train, columns=['isFraud']), BUCKET, f'{FEATURED_PREFIX}y_train.csv')
        save_csv_to_s3(X_test, BUCKET, f'{FEATURED_PREFIX}X_test_preprocessed.csv')
        
        if y_test is not None:
            save_csv_to_s3(pd.DataFrame(y_test, columns=['isFraud']), BUCKET, f'{FEATURED_PREFIX}y_test.csv')
        
        if train_ids is not None:
            save_csv_to_s3(pd.DataFrame(train_ids, columns=['TransactionID']), BUCKET, f'{FEATURED_PREFIX}train_ids.csv')
        if test_ids is not None:
            save_csv_to_s3(pd.DataFrame(test_ids, columns=['TransactionID']), BUCKET, f'{FEATURED_PREFIX}test_ids.csv')
        
        logger.info("✅ Preprocessed data saved successfully!")
        
        # ========================================
        # Preprocessing Summary
        # ========================================
        fraud_rate = y_train.mean() if y_train is not None else 0
        target_dist = y_train.value_counts().to_dict() if y_train is not None else {}
        
        logger.info("="*50)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*50)
        logger.info(f"Training samples: {X_train.shape[0]:,}")
        logger.info(f"Test samples: {X_test.shape[0]:,}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        logger.info(f"Target distribution (train): {target_dist}")
        logger.info(f"Fraud rate: {fraud_rate:.2%}")
        logger.info("✅ Data is ready for modeling!")
        
        return {
            "status": "success",
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "num_features": int(X_train.shape[1]),
            "fraud_rate": float(fraud_rate),
            "target_distribution": target_dist,
            "files_saved": [
                f"{FEATURED_PREFIX}X_train_preprocessed.csv",
                f"{FEATURED_PREFIX}y_train.csv",
                f"{FEATURED_PREFIX}X_test_preprocessed.csv",
                f"{FEATURED_PREFIX}y_test.csv",
                f"{FEATURED_PREFIX}train_ids.csv",
                f"{FEATURED_PREFIX}test_ids.csv"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e)
        }