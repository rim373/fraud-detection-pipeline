#!/usr/bin/env python3
"""
Test du webhook n8n
"""

import requests
import json
from datetime import datetime

N8N_WEBHOOK = "http://18.234.87.105:5678/webhook/fraud-test"

test_data = {
    "transaction_id": 99999,
    "amount": 2500.00,
    "risk_score": 85,
    "risk_level": "CRITICAL",
    "fraud_probability": 0.85,
    "card_type": "visa",
    "email_domain": "test.com",
    "device_type": "mobile",
    "risk_factors": "high_amount,suspicious_email",
    "timestamp": datetime.now().isoformat()
}

response = requests.post(N8N_WEBHOOK, json=test_data)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
