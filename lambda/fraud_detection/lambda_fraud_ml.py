import json
import boto3
import pandas as pd
from io import BytesIO
from datetime import datetime

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda de détection de fraude avec 8 règles business
    """
    try:
        # Récupérer le fichier S3
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        print(f"Processing: s3://{bucket}/{key}")
        
        # Charger le CSV
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
        
        print(f"Total transactions: {len(df)}")
        
        # Appliquer les règles de détection
        df['risk_score'] = 0
        df['risk_factors'] = ''
        
        # Règle 1: Montant élevé
        df.loc[df['TransactionAmt'] > 2000, 'risk_score'] += 35
        df.loc[df['TransactionAmt'] > 2000, 'risk_factors'] += 'very_high_amount,'
        
        df.loc[(df['TransactionAmt'] > 1000) & (df['TransactionAmt'] <= 2000), 'risk_score'] += 25
        df.loc[(df['TransactionAmt'] > 1000) & (df['TransactionAmt'] <= 2000), 'risk_factors'] += 'high_amount,'
        
        df.loc[(df['TransactionAmt'] > 500) & (df['TransactionAmt'] <= 1000), 'risk_score'] += 15
        df.loc[(df['TransactionAmt'] > 500) & (df['TransactionAmt'] <= 1000), 'risk_factors'] += 'medium_amount,'
        
        # Règle 2: Montant rond
        df.loc[df['TransactionAmt'] % 100 == 0, 'risk_score'] += 15
        df.loc[df['TransactionAmt'] % 100 == 0, 'risk_factors'] += 'round_amount,'
        
        # Règle 3: Email suspect
        suspicious_domains = ['anonymous', 'tempmail', 'guerrillamail', 'mailinator', '10minutemail']
        for domain in suspicious_domains:
            mask = df['P_emaildomain'].str.contains(domain, na=False, case=False)
            df.loc[mask, 'risk_score'] += 30
            df.loc[mask, 'risk_factors'] += 'suspicious_email,'
        
        # Règle 4: Email manquant
        df.loc[df['P_emaildomain'].isna(), 'risk_score'] += 20
        df.loc[df['P_emaildomain'].isna(), 'risk_factors'] += 'missing_email,'
        
        # Règle 5: Carte Discover
        df.loc[df['card4'] == 'discover', 'risk_score'] += 10
        df.loc[df['card4'] == 'discover', 'risk_factors'] += 'discover_card,'
        
        # Règle 6: Mobile + Montant élevé
        mask = (df['DeviceType'] == 'mobile') & (df['TransactionAmt'] > 1000)
        df.loc[mask, 'risk_score'] += 15
        df.loc[mask, 'risk_factors'] += 'mobile_high_amount,'
        
        # Règle 7: Informations manquantes
        missing_count = df[['addr1', 'card2', 'card5']].isna().sum(axis=1)
        df.loc[missing_count >= 2, 'risk_score'] += 20
        df.loc[missing_count >= 2, 'risk_factors'] += 'missing_data,'
        
        # Classification
        df['prediction'] = df['risk_score'].apply(lambda x: 'FRAUD' if x >= 45 else 'LEGITIMATE')
        
        df['risk_level'] = pd.cut(
            df['risk_score'],
            bins=[-1, 20, 40, 60, 80, 100],
            labels=['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        )
        
        df['fraud_probability'] = (df['risk_score'] / 100).clip(0, 1)
        df['model_version'] = 'business_rules_v1.0'
        
        # Nettoyer risk_factors
        df['risk_factors'] = df['risk_factors'].str.rstrip(',')
        
        # Sauvegarder les résultats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_key = f"fraud-predictions/predictions_{timestamp}.csv"
        
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        s3_client.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.getvalue()
        )
        
        fraud_count = len(df[df['prediction'] == 'FRAUD'])
        legitimate_count = len(df[df['prediction'] == 'LEGITIMATE'])
        
        print(f"✅ Detection completed:")
        print(f"   Total: {len(df)}")
        print(f"   Fraud: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
        print(f"   Legitimate: {legitimate_count}")
        print(f"   Output: s3://{bucket}/{output_key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Predictions completed successfully',
                'total_transactions': len(df),
                'fraud_count': fraud_count,
                'legitimate_count': legitimate_count,
                'fraud_rate': fraud_count / len(df),
                'output_file': f"s3://{bucket}/{output_key}"
            })
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }