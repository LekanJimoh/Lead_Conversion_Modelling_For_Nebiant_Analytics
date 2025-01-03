# Predictive Lead Scoring Model

## **Project Overview**
The Predictive Lead Scoring project aims to develop a machine learning model that prioritizes leads based on their likelihood of conversion (e.g., "High", "Medium", "Low"). This will help the sales team at Nebiant Analytics allocate resources efficiently and focus on high-value leads.

---

## **Objectives**
- Predict lead conversion likelihood.
- Provide actionable insights to improve lead management.
- Highlight the top factors influencing lead conversions.

---

## **Dataset Description**
The dataset contains the following key columns:

- **Demographic Details**: Location, Gender.
- **Lead Attributes**: Lead Source, Assigned Lead Manager.
- **Engagement Data**: Comments/Feedback sentiment scores.
- **Timing Data**: Days to start training.
- **Target Variable**: Lead Conversion Likelihood (Low, Medium, High).

---

## **Steps and Methodology**

### 1. **Data Preparation**
- Removed duplicates and irrelevant columns.
- Handled missing values.
- Cleaned and preprocessed text columns (e.g., Comments and Feedback).
- Encoded categorical variables using one-hot encoding and label encoding.
- Extracted time-to-start metrics from timing fields.

### 2. **Exploratory Data Analysis (EDA)**
- Analyzed demographic trends by location and gender.
- Examined lead conversion rates by source, timing, and program.
- Identified patterns in assigned vs. unassigned leads.

### 3. **Feature Engineering**
- Derived sentiment scores for comments and feedback.
- Created additional features for demographic and lead source trends.

### 4. **Model Development**
- **Models Used**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - XGBoost Classifier
- **Performance Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### 5. **Insights and Recommendations**
- Generated lead conversion scores and ranked leads.
- Explained key factors driving predictions using SHAP values.
- Recommended automating lead assignment to managers for better results.

---

## **Key Findings**
- Assigned leads have significantly higher conversion rates than unassigned leads.
- Sentiment scores from comments and feedback play a crucial role in predicting lead conversion likelihood.
- Timing data (e.g., days-to-start) is an important predictor of conversion.

---

## **Recommendations**
- Automate lead assignments to ensure no leads remain unassigned.
- Provide training for lead managers on follow-up strategies.
- Monitor key metrics such as "time to assign" and "time to first contact" to enhance efficiency.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - Data Processing: pandas, numpy
  - Visualization: matplotlib, seaborn
  - Machine Learning: scikit-learn, xgboost
  - Sentiment Analysis: TextBlob, VADER
- **Tools**:
  - Jupyter Notebook

---

## **How to Use**
1. Clone the repository.
2. Install required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to preprocess data, train the model, and evaluate results.
4. View insights and recommendations in the output.

---

## **Future Improvements**
- Incorporate more advanced NLP techniques for sentiment analysis.
- Use time-series analysis for better timing predictions.
- Develop a real-time dashboard for monitoring lead scores.

---

## **Acknowledgments**
Special thanks to the team at Nebiant Analytics for providing the dataset and project guidance.

