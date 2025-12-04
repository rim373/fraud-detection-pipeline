# 🧠 Step 2: Set Up SageMaker Studio or Notebook Instance

This step guides you through creating and configuring an Amazon SageMaker Notebook instance for model development, training, and experimentation.

---

## 1️⃣ Open SageMaker in AWS Console

1. Go to the **AWS Management Console**.  
2. Navigate to **Amazon SageMaker → Notebook instances**.  
3. Choose **Create notebook instance**.

---

## 2️⃣ Configure Notebook Instance Settings

| Parameter | Example / Recommended Value | Description |
|------------|-----------------------------|--------------|
| **Notebook instance name** | `xgboost` | Must be unique within your AWS Region (up to 63 alphanumeric characters, hyphens allowed). |
| **Notebook instance type** | `ml.t3.medium` | Balanced CPU and memory, suitable for lightweight development and testing. |
| **Platform identifier** | `Amazon Linux 2, JupyterLab 4` | Recommended environment for modern JupyterLab support. |
| **IAM role** | `SageMakerExecutionRole` | Must have S3 read/write permissions and SageMaker full access. |
| **Permissions policies to attach:** |  | - `AmazonS3FullAccess`<br>- `AmazonSageMakerFullAccess` |
| **Root access** | Enabled | Allows installing custom libraries if needed. |
| **Lifecycle configuration (optional)** | `init-setup` | Script to automatically install dependencies or clone repos on startup. |
| **VPC / Security groups** | Default | Leave default unless specific networking is required. |
| **Encryption key (optional)** | Default (AWS-managed) | Use a KMS key if your data requires encryption. |

---

## 3️⃣ Launch and Access the Notebook

1. Click **Create notebook instance**.
2. Wait until the status changes to **InService**.
3. Click **Open JupyterLab**.

