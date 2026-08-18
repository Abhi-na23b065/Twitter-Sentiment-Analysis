📌 Twitter Sentiment Analysis (1.6 Million Tweets — Sentiment140 Dataset)

A Machine Learning project using Logistic Regression and XGBoost to classify tweets into Positive or Negative sentiments.
This project is implemented in Google Colab, uses the Sentiment140 Kaggle Dataset, and includes full preprocessing, model training, saving, and evaluation.

📁 Dataset

The project uses the Sentiment140 dataset from Kaggle:

🔗 Dataset Link: https://www.kaggle.com/datasets/kazanova/sentiment140

Dataset Description

The dataset contains 1,600,000 labeled tweets with the following columns:

Column	Description
target	Sentiment (0 = Negative, 4 = Positive)
id	Tweet ID
date	Date of the tweet
flag	Query (not used)
user	Username
text	Tweet content

For modeling, only text and target columns were used.

🧹 Data Preprocessing

The preprocessing steps include:

Converting text to lowercase

Removing URLs

Removing @mentions

Removing special characters & punctuation

Removing extra spaces

Optional: Stemming (PorterStemmer)

Removing stopwords

TF-IDF vectorization

Since stemming on 1.6 million tweets is slow, optimized preprocessing was applied.

🤖 Machine Learning Models Used
1️⃣ Logistic Regression

Works well with TF-IDF features

Fast and efficient for text data

Achieved the highest accuracy in our tests

2️⃣ XGBoost Classifier

Gradient boosted decision tree model

Used to compare performance with Logistic Regression

Requires dense input → slower on large TF-IDF vectors

Performance was slightly lower due to the high dimensional text data

📊 Model Performance
Model	Training Accuracy	Test Accuracy
Logistic Regression	~79%	~78%
XGBoost	~84–88% (train)	~72–75% (test)

Conclusion:
Logistic Regression generalizes better on this dataset and feature space, while XGBoost tends to overfit TF-IDF text features.


📦 Saving the Trained Models

Models were saved using Python's pickle:

import pickle

filename = "trained_model.sav"
with open(filename, "wb") as file:
    pickle.dump(model, file)


Download model from Google Colab:

from google.colab import files
files.download("trained_model.sav")

📝 How to Run This Project
1️⃣ Upload the notebook to Google Colab

Open Colab → File → Upload Notebook

2️⃣ Install dependencies
pip install xgboost
pip install scikit-learn
pip install nltk

3️⃣ Download dataset using your Kaggle API token
!kaggle datasets download -d kazanova/sentiment140

4️⃣ Run preprocessing, training, and evaluation cells
5️⃣ Download the trained model for deployment

