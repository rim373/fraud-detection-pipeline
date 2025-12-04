#!/bin/bash

echo "🚀 Déploiement du pipeline complet"

# 1. Déployer les Lambda
cd lambda
./deploy.sh

# 2. Configurer S3
aws s3 mb s3://data-pipeline-$(date +%s) 2>/dev/null || true

# 3. Créer les crawlers Glue
cd ../aws/glue
./create_crawlers.sh

# 4. Démarrer n8n
cd ../../n8n
docker-compose up -d

echo "✅ Déploiement terminé !"