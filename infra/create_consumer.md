# ⚙️ Kafka Consumer & AWS Lambda Configuration

## 🎯 Purpose
The **Kafka Consumer** listens to the Kafka topic `transactions` and forwards incoming transaction data to an AWS **Lambda function (`consumer`)**, which stores these events in **Amazon S3 → /raw/**.  
This ensures every transaction streamed by the producer is persisted for later preprocessing.

---

## 🧩 COMPONENT 1 — AWS Lambda Setup (`consumer`)

### 1️⃣ Function Purpose
This Lambda receives one or multiple JSON records from EC2 (the Kafka consumer) and saves them into S3 as individual `.json` files under the `raw/` folder.

### 2️⃣ Configuration
| Parameter | Value |
|------------|--------|
| **Function name** | `consumer` |
| **Runtime** | Python 3.12 |
| **Memory** | 256 MB |
| **Timeout** | 60 seconds |
| **VPC** | ❌ *None (must stay outside a VPC)* |
| **Permissions** | `AWSLambdaBasicExecutionRole`, `AmazonS3FullAccess` *(or minimal `s3:PutObject`)* |
| **Bucket** | Example: `financial-fraud-project` |

---


