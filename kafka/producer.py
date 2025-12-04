#!/usr/bin/env python3
"""
Kafka Producer - Fraud Detection Pipeline
Lit train_light.csv et publie vers Kafka
"""

import csv
import json
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configuration
KAFKA_BROKER = '172.31.25.233:9092'
TOPIC_NAME = 'pipeline-data'
CSV_FILE = '../data/train_light.csv'
BATCH_SIZE = 5

def create_producer():
    """Crée et retourne un producer Kafka"""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=3
    )

def read_csv_and_publish():
    """Lit le CSV et publie vers Kafka"""
    producer = create_producer()
    
    try:
        with open(CSV_FILE, 'r') as f:
            reader = csv.DictReader(f)
            batch = []
            count = 0
            
            for row in reader:
                batch.append(row)
                count += 1
                
                if len(batch) >= BATCH_SIZE:
                    # Envoyer le batch
                    for record in batch:
                        future = producer.send(TOPIC_NAME, value=record)
                        try:
                            future.get(timeout=10)
                        except KafkaError as e:
                            print(f"❌ Erreur envoi : {e}")
                    
                    print(f"✅ Batch envoyé : {len(batch)} messages (Total: {count})")
                    batch = []
                    time.sleep(0.1)
            
            # Envoyer le dernier batch
            if batch:
                for record in batch:
                    producer.send(TOPIC_NAME, value=record)
                print(f"✅ Dernier batch : {len(batch)} messages")
        
        print(f"\n🎉 Total envoyé : {count} transactions")
        
    finally:
        producer.flush()
        producer.close()

if __name__ == "__main__":
    print("🚀 Démarrage du Producer Kafka")
    print(f"📡 Broker : {KAFKA_BROKER}")
    print(f"📊 Topic : {TOPIC_NAME}")
    print(f"📁 Fichier : {CSV_FILE}\n")
    
    read_csv_and_publish()