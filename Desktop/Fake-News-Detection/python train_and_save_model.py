# train_and_save_model.py (run this once)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data():
    data_dir = os.path.join(os.getcwd(), 'data')  # Relative Path
    train_csv_path = os.path.join(data_dir, "mediaeval-2015-trainingset.csv")
    test_csv_path = os.path.join(data_dir, "mediaeval-2015-testset.csv")

    try:
        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        return df_combined
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_and_save_model():
    df = load_data()
    if df is None:
        print("Failed to load data.")
        return

    if 'tweetText' not in df.columns or 'label' not in df.columns:
        print("Missing required columns.")
        return

    X = df['tweetText'].fillna('')
    y = df['label']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'))
    ])

    pipe.fit(X, y_encoded)
    
    joblib.dump(pipe, "fake_news_model.pkl")
    joblib.dump(dict(zip(encoder.transform(encoder.classes_), encoder.classes_)), "label_map.pkl")
    print("Model and label map saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
