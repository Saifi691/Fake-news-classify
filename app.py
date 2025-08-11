import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Detector")
st.markdown("Detect if a news article is **Fake** or **Real** with visualization.")

# User Input
user_input = st.text_area("✍️ Enter News Content Here", height=200)

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter news text to analyze.")
    else:
        # Transform & Predict
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        proba = model.decision_function(vec_input)

        # Label & Probability
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.markdown(f"## Prediction: {label}")

        # Probability Graph
        st.subheader("Prediction Confidence")
        fig, ax = plt.subplots()
        sns.barplot(x=["Fake", "Real"], y=[proba[0] * -1 if prediction == 0 else 0, proba[0] if prediction == 1 else 0], palette="coolwarm", ax=ax)
        ax.set_ylabel("Confidence Score")
        st.pyplot(fig)

        # Word Cloud
        st.subheader("Word Cloud")
        wc = WordCloud(width=800, height=400, background_color="white").generate(user_input)
        fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
