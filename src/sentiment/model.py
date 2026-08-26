"""
Sentiment analysis model for fashion trend comments.

This module:
1. Loads labeled comments from Excel
2. Cleans and tokenizes text (emoji handling, stopword removal)
3. Trains AdaBoost classifier with TF-IDF features
4. Evaluates with classification report
5. Saves model to disk
6. Predicts on unlabeled comments
"""

import os
import re
import warnings
import pandas as pd
import numpy as np
import emoji
import joblib
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a file into a pandas DataFrame."""
    return pd.read_excel(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the data for the machine learning model."""
    df = df.drop_duplicates(subset='comments', keep='first')
    
    # Drop the column at index 608 if it exists (from original code)
    if 608 in df.columns:
        df = df.drop(columns=608, axis=1)
    
    return df


def tokenize(text: str):
    """Tokenize the input text, converting to lowercase and removing stopwords."""
    if text is None:
        return None
    
    # Convert emoji to text
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Replace unrecognized emoji (left as ::) with space
    text = re.sub('::', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove English stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    return tokens


def split_data(df: pd.DataFrame):
    """Split the data into training and testing sets."""
    df2 = df[df['category'].notnull()]
    X = df2.comments
    y = df2.category
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model() -> GridSearchCV:
    """Creates a machine learning pipeline for multiclass classification."""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', AdaBoostClassifier())
    ])

    params = {
        'tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 60, 70]
    }   

    return GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=-1)


def evaluate_model(model, X_test, y_test) -> float:
    """Trains and evaluates a machine learning model with test data."""
    y_pred = model.predict(X_test)

    print('Classification Report:')
    print(classification_report(y_test, y_pred, zero_division=0))

    accuracy = (y_test == y_pred).mean()
    print(f'accuracy: {accuracy}')
    return accuracy


def save_model(model, filename: str) -> None:
    """Saves the model to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)


def predict_unlabelled_data(model, input_path: str = None, output_path: str = None) -> None:
    """Predict categories for unlabeled comments."""
    if input_path is None:
        input_path = PROJECT_ROOT / "data" / "posts_comments.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "posts_comments_predicted.csv"
    
    df_comments = pd.read_csv(input_path, sep=None, engine='python')
    df_comments['comments'] = df_comments['comments'].fillna('unknown')
    df_comments['category'] = model.predict(df_comments['comments'])
    df_comments.to_csv(output_path, sep=';', index=False)


def main():
    """Main execution: train and evaluate sentiment model."""
    # Setup paths
    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    
    train_path = data_dir / "labelled_comments_train.xlsx"
    model_path = models_dir / "sentiment_analysis_model.joblib"
    
    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        print("Please place labelled_comments_train.xlsx in the data/ directory")
        return
    
    print("Loading data...")
    df = load_data(str(train_path))
    df = clean_data(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    print("Building and training model...")
    model = build_model()
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    print("Saving model...")
    save_model(model, str(model_path))
    
    print("Predicting unlabeled data (if available)...")
    predict_unlabelled_data(model)
    
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
