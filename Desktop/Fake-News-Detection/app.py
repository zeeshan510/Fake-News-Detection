import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import os # Import the os module for path manipulation

# Load data (combining both training and test datasets for comprehensive training)
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__) # Gets the directory of the current script (app.py)
    
    train_csv_path = os.path.join(base_dir, "Fake-News-Detection", "mediaeval-2015-trainingset.csv")
    test_csv_path = os.path.join(base_dir, "Fake-News-Detection", "mediaeval-2015-testset.csv")
    
    try:
        # Load training data
        df_train = pd.read_csv(train_csv_path)
        
        # Load test data
        df_test = pd.read_csv(test_csv_path)
        
        # Concatenate both dataframes to use all available data for training
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        
        return df_combined
    except FileNotFoundError as e:
        st.error(f"Error: One of the data files was not found. Please ensure '{e.filename}' is in the 'Fake-News-Detection' subfolder.")
        st.stop() # Stop the app if a file is not found
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

# Train the model
@st.cache_resource
def train_model(df):
    # Ensure the DataFrame has the expected columns
    if 'tweetText' not in df.columns or 'label' not in df.columns:
        st.error("Error: The combined CSV data must contain 'tweetText' and 'label' columns.")
        st.stop()
        
    X = df['tweetText']
    y = df['label']
    
    # Handle potential missing values in text data
    X = X.fillna('')

    encoder = LabelEncoder()
    try:
        y_encoded = encoder.fit_transform(y)
    except ValueError as e:
        st.error(f"Error encoding labels: {e}. Make sure 'label' column contains valid categories.")
        st.stop()

    st.write("Training Data Label Distribution (Combined Dataset):")
    st.write(df['label'].value_counts()) # Display the distribution of labels for the combined dataset
    
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')) # Added class_weight='balanced'
    ])
    
    try:
        pipe.fit(X, y_encoded)
    except Exception as e:
        st.error(f"Error training the model: {e}")
        st.stop()
        
    label_map = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
    return pipe, label_map

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector using Machine Learning")
st.write("Check if a news article is **Fake**, **Real**, or **Humor** by pasting it below.")

# Input box
user_input = st.text_area("✍️ Enter the news content here:")

# Load and train model
# These calls will now correctly look for the files in the specified subfolder
df = load_data()
model, label_map = train_model(df)

if st.button("🔍 Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some news content first.")
    else:
        try:
            prediction = model.predict([user_input])[0]
            label = label_map[prediction]
            if label == 'fake':
                st.error("❌ This news seems to be **FAKE**.")
            elif label == 'real':
                st.success("✅ This news seems to be **REAL**.")
            else:
                st.info("😄 This news might be **HUMOR**.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("🔧 Built with `Scikit-learn` and `Streamlit`")
