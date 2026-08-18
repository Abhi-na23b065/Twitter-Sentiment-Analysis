# Twitter Sentiment Analysis

A Machine Learning project that performs **sentiment analysis on Twitter data** using **Natural Language Processing (NLP)** and machine learning algorithms.

The project uses the **Sentiment140 dataset containing 1.6 million labeled tweets** and classifies tweets into two sentiment categories: **Positive** and **Negative**. The complete workflow includes data preprocessing, text cleaning, TF-IDF feature extraction, model training, evaluation, model comparison, and deployment using saved machine learning models.

## Features

### Data Analysis

* Load the Sentiment140 dataset
* Analyze dataset shape and structure
* Check missing values
* Analyze sentiment distribution
* Convert sentiment labels into binary classes
* Explore tweet text data

### Text Preprocessing

The following NLP preprocessing techniques are applied:

* Convert text to lowercase
* Remove non-alphabetic characters
* Remove stopwords
* Apply Porter Stemming
* Clean tweet content
* Prepare processed text for machine learning

The processed tweet text is stored in a separate `stemmed_content` column.

### TF-IDF Vectorization

**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert the processed tweet text into numerical feature vectors.

This allows machine learning algorithms to identify important words and patterns within tweets.

### Logistic Regression

Logistic Regression is used as the primary classification model.

The model is trained using TF-IDF features and predicts whether a tweet is:

```text
0 → Negative
1 → Positive
```

Logistic Regression provides strong performance on the high-dimensional sparse TF-IDF text representation.

### XGBoost

**XGBoost Classifier** is implemented as a second machine learning model to compare its performance with Logistic Regression.

The project analyzes:

* Training accuracy
* Testing accuracy
* Classification report
* Confusion matrix
* Overfitting gap

The comparison demonstrates that Logistic Regression generalizes better for this particular sparse TF-IDF text representation.

## Dataset

The project uses the **Sentiment140 dataset** from Kaggle.

The dataset contains **1,600,000 labeled tweets** with the following columns:

```text
target
id
date
flag
user
text
```

For model training, the project primarily uses:

```text
text
target
```

The original sentiment labels are:

```text
0 → Negative
4 → Positive
```

The positive label `4` is converted to `1` for binary classification.

## System Flow

```text
                    Twitter Sentiment Analysis
                              |
                              v
                    Sentiment140 Dataset
                              |
                              v
                       Data Cleaning
                              |
                              v
                    Text Preprocessing
                              |
                              v
                     Porter Stemming
                              |
                              v
                      TF-IDF Vectorizer
                              |
                +-------------+-------------+
                |                           |
                v                           v
        Logistic Regression             XGBoost
                |                           |
                v                           v
        Model Evaluation             Model Evaluation
                |                           |
                +-------------+-------------+
                              |
                              v
                       Model Comparison
                              |
                              v
                    Best Model Selection
                              |
                              v
                    Save Model + Vectorizer
                              |
                              v
                     New Tweet Prediction
```

## Model Evaluation

The project evaluates both models using training and testing accuracy.

| Model               | Training Accuracy | Test Accuracy |
| ------------------- | ----------------: | ------------: |
| Logistic Regression |              ~79% |          ~78% |
| XGBoost             |           ~84–88% |       ~72–75% |

The results show that **Logistic Regression performs better on the test data and generalizes better**, while XGBoost shows a larger gap between training and testing performance.

The project also generates:

* Classification Report
* Confusion Matrix
* Training Accuracy
* Testing Accuracy
* Overfitting Gap
* Model Accuracy Comparison

## Deployment

The trained models and TF-IDF vectorizer are saved using **Pickle**.

Saved files include:

```text
saved_models/
│
├── logistic_regression_model.pkl
├── xgboost_model.pkl
└── tfidf_vectorizer.pkl
```

Saving the vectorizer is important because new tweets must undergo the same feature transformation used during model training.

The deployment section loads the saved Logistic Regression model and TF-IDF vectorizer and predicts the sentiment of new tweets.

Example:

```text
Tweet:
"I absolutely love this! Best thing ever, feeling amazing today."

Prediction:
Positive
```

The deployment pipeline also provides a prediction confidence score.

## Technology Stack

* **Language:** Python
* **Environment:** Google Colab / Jupyter Notebook
* **NLP:** NLTK
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Model:** Logistic Regression
* **Model:** XGBoost
* **Feature Extraction:** TF-IDF
* **Visualization:** Matplotlib, Seaborn
* **Model Serialization:** Pickle

## Project Structure

```text
Twitter-Sentiment-Analysis/
│
├── Twitter_Final.ipynb
├── saved_models/
│   ├── logistic_regression_model.pkl
│   ├── xgboost_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── README.md
```

## Notebook Workflow

```text
1. Install Dependencies
        ↓
2. Download Sentiment140 Dataset
        ↓
3. Load Dataset
        ↓
4. Data Exploration
        ↓
5. Convert Sentiment Labels
        ↓
6. Text Preprocessing
        ↓
7. Porter Stemming
        ↓
8. Train-Test Split
        ↓
9. TF-IDF Vectorization
        ↓
10. Logistic Regression
        ↓
11. XGBoost
        ↓
12. Model Evaluation
        ↓
13. Confusion Matrix
        ↓
14. Model Comparison
        ↓
15. Save Models
        ↓
16. Deployment Testing
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Twitter-Sentiment-Analysis
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk xgboost kaggle
```

### 3. Open the Notebook

Open:

```text
Twitter_Final.ipynb
```

using **Google Colab, Jupyter Notebook, or VS Code**.

### 4. Download the Dataset

The notebook downloads the Sentiment140 dataset using the Kaggle API.

### 5. Run the Notebook

Execute the cells sequentially to perform:

* Data preprocessing
* Model training
* Evaluation
* Model comparison
* Model saving
* New tweet prediction

## Key Learning Outcomes

* Worked with a **1.6 million tweet NLP dataset**.
* Performed text preprocessing on large-scale Twitter data.
* Implemented stemming and stopword removal.
* Converted text into numerical features using TF-IDF.
* Implemented Logistic Regression for sentiment classification.
* Implemented XGBoost for model comparison.
* Evaluated models using accuracy and classification reports.
* Generated confusion matrices.
* Analyzed model overfitting.
* Saved trained models using Pickle.
* Built a basic deployment/inference pipeline for new tweets.

## Future Improvements

* Add a web interface using **Streamlit**.
* Deploy the model using **FastAPI or Flask**.
* Add real-time Twitter/X data collection.
* Improve preprocessing using lemmatization.
* Experiment with advanced NLP models such as BERT.
* Add Precision, Recall, F1-Score, and ROC-AUC comparisons.
* Build a dashboard for visualizing sentiment trends.
* Support multi-class sentiment analysis.

## Conclusion

This project demonstrates a complete **Natural Language Processing and Machine Learning pipeline** for Twitter sentiment classification. Using a large-scale dataset of 1.6 million tweets, the project performs text preprocessing, TF-IDF feature extraction, classification using Logistic Regression and XGBoost, model evaluation, comparison, and deployment-based prediction.

The project provides practical experience in **NLP, text preprocessing, feature engineering, supervised machine learning, model evaluation, and model deployment**, making it a strong machine learning project for a portfolio or resume.
