
---

## 🔵 `PREPROCESS_CONFIGURATION.md`

```markdown
# ⚙️ Preprocessing Lambda Configuration

## 🎯 Purpose
The **Preprocess Lambda** cleans raw JSON transaction data from S3, fills missing values, preserves class balance, splits it into train (70%) and test (30%), and stores the results in S3/processed/.

---

## 🧩 Setup

### Lambda Runtime
- **Runtime:** Python 3.12  
- **Layer:** `AWSSDKPandas-Python312`
- **Timeout:** ≥ 60 seconds  
- **Memory:** 512 MB  
- **Bucket:** same as used in the consumer Lambda (e.g. `financial-fraud-project`)

---
