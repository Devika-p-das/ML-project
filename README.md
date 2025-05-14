# ML-project

## 🔍 Network Attack Classification - Machine Learning Project

## 📌 Project Description
This project focuses on classifying network traffic based on attack types using supervised machine learning models. It includes data exploration, preprocessing,Feature Engineering, SMOTE for class balancing, and a modeling pipeline with performance comparison across several classification algorithms.

**📂 Project Structure**:
     * Data Loading
     * Data Exploration
     * Data Visualization
     * Data Cleaning
     * Data Splitting
     * Model Building
     * Model Evaluation


4.  **Model Used:** Trained and compared several machine learning models:
    * XGBoost Classifier
    * Logistic Regression
    * Decision Tree
    * Random Forest
    * Gradient Boosting
    * K-Nearest Neighbors (KNN)
5.  **Model Evaluation:** Evaluated the models using metrics such as:
    * Accuracy
    * Precision
    * Recall
    * F1-score
    * ROC AUC
    * Confusion Matrix
6.  **Model Saving:** The best-performing model was saved for potential future use.

## 🛠 Tech Stack / Tools Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Jupyter Notebook
* XGBoost
* imblearn (for SMOTE)

## Results

The following table summarizes the performance of the trained models:

| Model                  | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ---------------------- | -------- | --------- | ------ | -------- | ------- |
| XGBoost Classifier     | 0.999    | 0.999     | 0.999  | 0.999    | 0.999   |
| Logistic Regression    | 0.947    | 0.949     | 0.947  | 0.947    | 0.987   |
| Decision Tree          | 0.998    | 0.998     | 0.998  | 0.998    | 0.998   |
| Random Forest          | 0.999    | 0.999     | 0.999  | 0.999    | 0.999   |
| Gradient Boosting      | 0.999    | 0.999     | 0.999  | 0.999    | 0.999   |
| SVM                    | 0.988    | 0.988     | 0.988  | 0.988    | 0.997   |
| K-Nearest Neighbors    | 0.996    | 0.996     | 0.996  | 0.996    | 0.999   |

✅ **Best Model: XGBoost**




