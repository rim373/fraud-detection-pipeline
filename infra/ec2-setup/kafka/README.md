# EC2 Setup — Fraud Detection Lab

## 1. Launch EC2 Instance
- Region: `us-east-1`
- Name: `fraud-ec2-kafka`
- AMI: Ubuntu 22.04 LTS
- Type: t3.medium
- IAM Role: `LabRole`
- Security Group ports:
  - 22 (SSH) from your IP
  - 2181 (Zookeeper)
  - 9092 (Kafka)
  - 3000 (Grafana)
- Keypair: your lab key (e.g., `labsuser.pem`)
- Storage: 20 GB

## 2. Connect via SSH
```bash
ssh -i ~/Downloads/labsuser.pem ubuntu@<EC2_PUBLIC_IP>
