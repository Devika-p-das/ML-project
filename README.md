# 💻 Network Attack Classification - Machine Learning Project

##  Data Description

The RT-IoT2022, a proprietary dataset derived from a real-time IoT infrastructure, is introduced as a comprehensive resource integrating a diverse range of IoT devices and sophisticated network attack methodologies. This dataset encompasses both normal and adversarial network behaviours, providing a general representation of real-world scenarios. Incorporating data from IoT devices such as ThingSpeak-LED, Wipro-Bulb,and MQTT-Temp, as well as simulated attack scenarios involving Brute-Force SSH attacks, DDoS attacks using Hping and Slowloris, and Nmap patterns, RT-IoT2022 offers a detailed perspective on the complex nature of network traffic. The bidirectional attributes of network traffic are meticulously captured using the Zeek network monitoring tool and the Flowmeter plugin. Researchers can leverage the RT-IoT2022 dataset to advance the capabilities of Intrusion Detection Systems (IDS), fostering the development of robust and adaptive security solutions for real-time IoT networks.

---

## 🔍 Project Description

This project focuses on classifying network traffic based on attack types using supervised machine learning models. The workflow includes data exploration, preprocessing, feature engineering, SMOTE for class balancing, and a full modeling pipeline with performance comparisons across several classification algorithms.

---

## 📁 Project Structure

1. Data Loading  
2. Data Exploration  
3. Data Visualization  
4. Data Cleaning  
5. Data Splitting  
6. Model Building  
7. Model Evaluation  
 

---

## 🧠 Models Used

Trained and compared several machine learning models:

- XGBoost Classifier  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- K-Nearest Neighbors (KNN)  

---

## 📊 Model Evaluation

Models were evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC AUC  
- Confusion Matrix  

---



## 🛠 Tech Stack / Tools Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- imbalanced-learn (SMOTE)  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

## 📈 Results

| Model                | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.214587 | 0.892726  | 0.214587 | 0.257776  | 0.906678 |
| Decision Tree        | 0.995045 | 0.995319  | 0.995045 | 0.995147  | 0.964636 |
| Random Forest        | 0.996426 | 0.996528  | 0.996426 | 0.996451  | 0.985362 |
| K-Nearest Neighbors  | 0.986680 | 0.989769  | 0.986680 | 0.987941  | 0.960951 |
| XGBoost              | 0.996832 | 0.996898  | 0.996832 | 0.996842  | 0.999868 |
| Gradient Boosting    | 0.995695 | 0.995903  | 0.995695 | 0.995762  | 0.999814 |

---

✅ **Best Model: XGBoost Classifier**



**Data Source:  UCI Machine Learning Repository**
