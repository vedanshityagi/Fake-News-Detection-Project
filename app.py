import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰"
)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


# Frontend
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check whether it is real or fake.")

news = st.text_area(
    "Enter News Article:",
    height=200,
    placeholder="Paste your news article here..."
)

if st.button("🔍 Check News"):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:
        cleaned_news = clean_text(news)

        news_tfidf = vectorizer.transform([cleaned_news])

        prediction = model.predict(news_tfidf)[0]

        if prediction == 0:
            st.error("❌ FAKE NEWS")

        else:
            st.success("✅ REAL NEWS")