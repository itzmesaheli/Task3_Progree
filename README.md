# Intelligent Multi-Class Text Sentiment Classifier

Progree AI Internship - Task 2

A multi-class (positive / neutral / negative) sentiment classifier built with Python and Scikit-Learn.

## Pipeline
1. Text cleaning (lowercase, remove URLs/mentions/punctuation, expand n't)
2. Stop-word removal (NLTK), keeping negations such as "not" and "no"
3. Lemmatization (NLTK WordNet, POS-aware)
4. TF-IDF features (unigrams + bigrams)
5. Models compared: Logistic Regression, Linear SVM, Multinomial Naive Bayes
6. Evaluation: per-class F1, macro F1, weighted F1, confusion matrix

## Setup
```
pip install -r requirements.txt
```

## Run
```
# Real dataset (example: Twitter US Airline Sentiment from Kaggle)
python sentiment_classifier.py --data Tweets.csv --text-col text --label-col airline_sentiment

# Classify a new sentence with the saved model
python sentiment_classifier.py --predict "The service was awful"
```

## Outputs (in `results/`)
- `model_comparison.csv` - accuracy, macro F1, weighted F1 for every model
- `classification_report.txt` - precision / recall / F1 per class (best model)
- `confusion_matrix.png`, `f1_per_class.png`
- `metrics.json`, `best_model.joblib`

