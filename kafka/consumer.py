#!/usr/bin/env python3
"""
Kafka Consumer - Fraud Detection Pipeline
Consomme depuis Kafka et upload vers S3
"""

import json
import csv
import boto3
from kafka import KafkaConsumer
from datetime import datetime
from io import StringIO

# Configuration
KAFKA_BROKER = '172.31.25.233:9092'
TOPIC_NAME = 'pipeline-data'
GROUP_ID = 'consumer-etl-group'
S3_BUCKET = 'data-pipeline-1764670683'
S3_PREFIX = 'processed-data/'

def create_consumer():
    """Crée et retourne un consumer Kafka"""
    return KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_BROKER,
        group_id=GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True
    )

def upload_to_s3(data, filename):
    """Upload data vers S3"""
    s3_client = boto3.client('s3')
    
    # Convertir en CSV
    output = StringIO()
    if data:
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    # Upload
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=f"{S3_PREFIX}{filename}",
        Body=output.getvalue()
    )
    print(f"✅ Uploaded: s3://{S3_BUCKET}/{S3_PREFIX}{filename}")

def consume_and_process():
    """Consomme depuis Kafka et traite"""
    consumer = create_consumer()
    buffer = []
    BUFFER_SIZE = 100
    
    try:
        print("👂 En écoute des messages...")
        
        for message in consumer:
            data = message.value
            buffer.append(data)
            
            if len(buffer) >= BUFFER_SIZE:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"train_light_processed_{timestamp}.csv"
                upload_to_s3(buffer, filename)
                print(f"📦 Buffer uploadé : {len(buffer)} records")
                buffer = []
                
    except KeyboardInterrupt:
        print("\n⏹️  Arrêt du consumer...")
        if buffer:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"train_light_processed_{timestamp}.csv"
            upload_to_s3(buffer, filename)
    finally:
        consumer.close()

if __name__ == "__main__":
    print("🚀 Démarrage du Consumer Kafka")
    print(f"📡 Broker : {KAFKA_BROKER}")
    print(f"📊 Topic : {TOPIC_NAME}")
    print(f"🪣 S3 Bucket : {S3_BUCKET}\n")
    
    consume_and_process()