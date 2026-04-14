# 🔐 Machine Learning-Based Intrusion Detection System (IDS)

## 📌 Project Overview

This project implements a **Machine Learning-based Intrusion Detection System (IDS)** for analyzing IoT network traffic. It applies feature engineering techniques and compares multiple ML models to classify network traffic as **Normal (0)** or **Malicious (1)**.

The system evaluates different classifiers and identifies the best-performing model based on accuracy.
PPT LINK : https://docs.google.com/presentation/d/1CkyBb7zOn2pZg3jsL0u7Vidb_DEGhf4d/edit?usp=sharing&ouid=101386664084771135450&rtpof=true&sd=true
---

## 🚀 Features

* Data preprocessing and cleaning
* Protocol encoding and IP feature extraction
* Behavioral feature engineering
* Feature scaling using StandardScaler
* Multiple ML model comparison
* Model performance evaluation using:

  * Accuracy
  * Classification Report
  * Confusion Matrix

---

## 📂 Dataset

* File: `philips.csv`
* Encoding: `latin1`
* Invalid rows skipped automatically
* Since the dataset does not contain labels, synthetic labels are generated for demonstration purposes.

⚠ Note: For real-world deployment, labeled intrusion data should be used.

---

## 🧠 Feature Engineering

The following features are extracted:

* **Protocol_encoded** – Encoded protocol type
* **Length** – Packet length (numeric conversion)
* **Time_delta** – Time difference between packets
* **Source_encoded** – Encoded source IP prefix
* **Dest_encoded** – Encoded destination IP prefix
* **Is_MDNS** – mDNS protocol flag
* **Is_DHCP** – DHCP protocol flag
* **Is_NTP** – NTP protocol flag

---

## 🤖 Machine Learning Models Used

The system compares the following classifiers:

* Random Forest
* Logistic Regression
* Support Vector Machine (SVM)
* XGBoost
* Neural Network (MLPClassifier)

Each model is trained and evaluated on a 70-30 train-test split.

---

## 📊 Evaluation Metrics

* Accuracy Score
* Classification Report (Precision, Recall, F1-score)
* Confusion Matrix

The best-performing model is selected based on highest accuracy.

---

## 🛠 Installation

Make sure you have Python 3.8+ installed.

Install required dependencies:

```bash
pip install pandas numpy scikit-learn xgboost
```

---

## ▶️ How to Run

1. Place `philips.csv` in the project directory.
2. Run the script:

```bash
python IDS.py
```

3. View model accuracies and performance metrics in the terminal.

---

## 📈 Output

The script will:

* Train all models
* Print performance metrics
* Display confusion matrices
* Identify the best-performing model

---

## 🔮 Future Improvements

* Use real labeled intrusion datasets (e.g., CICIDS)
* Improve feature engineering
* Handle class imbalance using SMOTE
* Hyperparameter tuning
* Deploy as a real-time IDS
* Add visualization dashboards

---

## 📚 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

---

## 👩‍💻 Author

Suhani Goyal
B.Tech – Computer Science and Engineering

---
