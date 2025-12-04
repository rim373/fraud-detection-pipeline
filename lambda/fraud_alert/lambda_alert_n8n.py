import json
import boto3
import urllib3
import os
from datetime import datetime

s3_client = boto3.client('s3')
http = urllib3.PoolManager()
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', '')

def lambda_handler(event, context):
    """
    Lambda d'alerte - Envoie les fraudes critiques à n8n
    """
    try:
        import pandas as pd
        
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        print(f"Processing: s3://{bucket}/{key}")
        
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
        
        # Filtrer fraudes critiques
        critical = df[
            (df['prediction'] == 'FRAUD') & 
            (df['risk_level'].isin(['CRITICAL', 'HIGH']))
        ]
        
        print(f"Found {len(critical)} critical/high frauds")
        
        if len(critical) == 0:
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'No critical frauds'})
            }
        
        emails_sent = 0
        
        # Envoyer à n8n (max 10 alertes)
        for idx, row in critical.head(10).iterrows():
            fraud_data = {
                'transaction_id': int(row['TransactionID']),
                'amount': float(row['TransactionAmt']),
                'risk_score': int(row['risk_score']),
                'risk_level': str(row['risk_level']),
                'fraud_probability': float(row['fraud_probability']),
                'card_type': str(row.get('card4', '')),
                'email_domain': str(row.get('P_emaildomain', '')),
                'device_type': str(row.get('DeviceType', '')),
                'risk_factors': str(row.get('risk_factors', '')),
                'timestamp': datetime.now().isoformat()
            }
            
            if N8N_WEBHOOK_URL:
                try:
                    response = http.request(
                        'POST',
                        N8N_WEBHOOK_URL,
                        body=json.dumps(fraud_data).encode('utf-8'),
                        headers={'Content-Type': 'application/json'}
                    )
                    print(f"n8n webhook response: {response.status}")
                    if response.status == 200:
                        emails_sent += 1
                except Exception as e:
                    print(f"n8n webhook error: {str(e)}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Alerts sent',
                'critical_frauds': len(critical),
                'alerts_sent': emails_sent
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }