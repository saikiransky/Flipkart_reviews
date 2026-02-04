import streamlit as st
import pickle

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Flipkart Reviews",
    page_icon="🛒",
    layout="centered"
)

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    with open("Flipkart_SVC.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ------------------ Header ------------------
st.title("Flipkart Reviews Sentiment Analysis")
st.write("Enter a product review and predict whether it's Positive or Negative.")

# ------------------ Input ------------------
user_input = st.text_area(
    "Review Text",
    placeholder="Example: Good quality product...",
    height=120
)

# ------------------ Prediction ------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        text = user_input.strip().lower()

        # Rule phrases
        positive_phrases = [
            "i like", "i love", "very good", "excellent",
            "awesome", "perfect", "good quality",
            "high quality", "worth it"
        ]
        negative_phrases = [
            "don't like", "do not like", "not good",
            "waste of", "worst", "very bad",
            "poor quality", "not worth", "bad quality"
        ]

        with st.spinner("Analyzing review..."):
            if any(p in text for p in negative_phrases):
                prediction = "Negative"
                confidence = 0.95
            elif any(p in text for p in positive_phrases):
                prediction = "Positive"
                confidence = 0.95
            else:
                prediction = model.predict([user_input])[0]
                score = model.decision_function([user_input])[0]
                confidence = min(1.0, abs(score) / 3)

        # ------------------ Result ------------------
        st.subheader("Result")
        if prediction == "Positive":
            st.success(f"Sentiment: {prediction}")
        else:
            st.error(f"Sentiment: {prediction}")

        