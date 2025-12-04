# 🚨 Fraud Detection Pipeline - Real-Time ML System

Pipeline complet de détection de fraude en temps réel utilisant AWS, Kafka, Lambda, Grafana et n8n.

![Architecture](docs/pipeline.png)

## 🎯 Caractéristiques

- **Ingestion temps réel** : Kafka (3 partitions)
- **Détection ML** : 8 règles business sophistiquées
- **Stockage** : S3 Data Lake
- **Analytics** : Glue + Athena + Grafana
- **Alertes** : n8n automation + Email notifications
- **Performance** : 15,960 transactions analysées en 7 minutes

## 📊 Résultats

- **Volume traité** : 42,525 transactions → 15,960 analysées
- **Fraudes détectées** : 258 (1.6%)
- **Alertes critiques** : ~50 (risk_score ≥ 70)
- **Taux de réussite n8n** : 76%

## 🏗️ Architecture
```
Producer → Kafka → Consumer → S3 processed-data/
                                    ↓
                          Lambda Detection (ML)
                                    ↓
                        S3 fraud-predictions/
                              ↓         ↓
                        Lambda Alert   Glue Crawler
                              ↓         ↓
                        n8n Webhook   Athena
                              ↓         ↓
                    Email/Slack    Grafana
```

## 🚀 Quick Start

### Prérequis

- AWS Account (Learner Lab ou standard)
- Python 3.9+
- Docker
- AWS CLI configuré

### Installation
```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/fraud-detection-pipeline.git
cd fraud-detection-pipeline

# 2. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos paramètres AWS

# 3. Déployer l'infrastructure
./scripts/setup_infrastructure.sh

# 4. Déployer les Lambda
cd lambda
./deploy.sh

# 5. Lancer Kafka
cd ../kafka
python producer.py &
python consumer.py &

# 6. Démarrer n8n
cd ../n8n
docker-compose up -d

# 7. Importer le dashboard Grafana
# Importer grafana/dashboards/fraud_detection_dashboard.json
```

## 📖 Documentation

- [Architecture détaillée](docs/ARCHITECTURE.md)
- [Guide d'installation](docs/INSTALLATION.md)
- [Configuration](docs/CONFIGURATION.md)

## 🧪 Tests
```bash
# Test du pipeline complet
python tests/test_pipeline.py

# Test du webhook n8n
python tests/test_n8n_webhook.py
```

## 📈 Dashboards

- **Grafana** : http://your-grafana-ip:3000
- **n8n** : http://your-n8n-ip:5678

## 🛠️ Technologies

- **Streaming** : Apache Kafka 3.5.1
- **Cloud** : AWS (EC2, Lambda, S3, Glue, Athena)
- **ML** : Python + Règles Business
- **Visualisation** : Grafana 10.x
- **Automation** : n8n
- **IaC** : CloudFormation + Bash

