# ⚙️ Step 1 — Launch a New EC2 Instance

## 1️⃣ Go to AWS Console
- Navigate to **EC2 → Instances → Launch Instance**

## 2️⃣ Instance Configuration

| Setting | Value / Description |
|----------|--------------------|
| **Name** | `fraud-training-ec2` |
| **AMI** | Ubuntu 22.04 LTS |
| **Instance Type** | `t3.medium` or `t3.large` *(recommended for XGBoost CPU)* |
| **Key Pair** | Choose an existing key pair or **create a new one** (used for SSH access) |

## 3️⃣ Network Settings
- ✅ **Allow SSH (port 22)**  
- *(Optional)* Attach to the **same VPC/Subnet** as your main EC2 instance if needed

## 4️⃣ Storage
- Set **Storage Size:** 20–30 GB

## 5️⃣ Launch
- Click **Launch Instance**  
- Wait until the status changes to **Running**

---

### ✅ Notes
- After launching, you can connect via SSH:  
  ```bash
  ssh -i "your-key.pem" ubuntu@<EC2-PUBLIC-IP>


