# 💻 Network Attack Classification - Machine Learning Project

##  Data Description

The RT-IoT2022, a proprietary dataset derived from a real-time IoT infrastructure, is introduced as a comprehensive resource integrating a diverse range of IoT devices and sophisticated network attack methodologies. This dataset encompasses both normal and adversarial network behaviours, providing a general representation of real-world scenarios. Incorporating data from IoT devices such as ThingSpeak-LED, Wipro-Bulb,and MQTT-Temp, as well as simulated attack scenarios involving Brute-Force SSH attacks, DDoS attacks using Hping and Slowloris, and Nmap patterns, RT-IoT2022 offers a detailed perspective on the complex nature of network traffic. The bidirectional attributes of network traffic are meticulously captured using the Zeek network monitoring tool and the Flowmeter plugin. Researchers can leverage the RT-IoT2022 dataset to advance the capabilities of Intrusion Detection Systems (IDS), fostering the development of robust and adaptive security solutions for real-time IoT networks.

---

## 🔍 Project Description

This project focuses on classifying IoT network traffic based on attack types using supervised machine learning models. The workflow includes data exploration, preprocessing, feature engineering, SMOTE for class balancing, and a complete modeling pipeline with performance comparisons across multiple classification algorithms.


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

| Model               | Accuracy  | Precision | Recall   | F1 Score | ROC AUC  |
|---------------------|-----------|-----------|----------|----------|----------|
| Logistic Regression | 0.177307  | 0.914390  | 0.177307 | 0.243616 | 0.739125 |
| Decision Tree       | 0.995858  | 0.996111  | 0.995858 | 0.995953 | 0.961476 |
| Random Forest       | 0.996589  | 0.996783  | 0.996589 | 0.996654 | 0.985502 |
| KNN                 | 0.985705  | 0.988846  | 0.985705 | 0.986982 | 0.957713 |
| XGBoost             | 0.996426  | 0.996590  | 0.996426 | 0.996487 | 0.999834 |
| Gradient Boosting   | 0.995533  | 0.996163  | 0.995533 | 0.995775 | 0.999853 |


---

✅ **Best Model: XGBoost Classifier**



**Data Source:  UCI Machine Learning Repository**
