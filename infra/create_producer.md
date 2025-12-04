# ⚙️ Kafka Producer Configuration

## 🎯 Purpose
The **Kafka Producer** streams transaction data from a local CSV file into the Kafka topic `transactions`.  
This component simulates real-time financial transactions to feed the fraud detection pipeline.

---

## 🧩 Environment Setup
- activate the virtual env
- create a new topic 
- update the kafka_producer.py (with the new topic name )
- Ensure the kafka is running 
- run puthon3 kafka_producer.py



### Requirements
- Python ≥ 3.12  
- Libraries:  
  ```bash
  pip install kafka-python pandas