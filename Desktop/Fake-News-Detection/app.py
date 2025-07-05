import streamlit as st
import joblib

@st.cache_resource
def load_model():
    try:
        model = joblib.load("fake_news_model.pkl")
        label_map = joblib.load("label_map.pkl")
        return model, label_map
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, label_map = load_model()

st.title("📰 Fake News Detector using Machine Learning")
st.markdown("Check if a news article is **Fake**, **Real**, or **Humor** by pasting it below.")

news_content = st.text_area("✍️ Enter the news content here:")

if st.button("🔍 Detect"):
    if not news_content.strip():
        st.warning("Please enter some news content.")
    elif model is None or label_map is None:
        st.error("Model not loaded. Train & save the model first.")
    else:
        try:
            prediction_encoded = model.predict([news_content])[0]
            predicted_label = label_map[prediction_encoded]
            st.success(f"Prediction: **{predicted_label.upper()}**")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.markdown("---")
st.markdown("🛠️ Built with [Scikit-learn](https://scikit-learn.org/) and [Streamlit](https://streamlit.io/)")
