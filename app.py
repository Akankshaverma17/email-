import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Spam Email Detection", layout="wide")

st.title("📧 Spam Email Detection App")

uploaded_file = st.file_uploader("Upload your email dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    required_cols = {"subject", "body", "label"}
    if not required_cols.issubset(df.columns):
        st.error("Dataset must contain subject, body, and label columns")
    else:
        # Combine subject + body
        df["text"] = df["subject"] + " " + df["body"]

        X = df["text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", MultinomialNB())
        ])

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        st.success(f"✅ Model Accuracy: {accuracy:.2f}")

        df["prediction"] = model.predict(X)

        st.subheader("🔍 Filter Emails")

        filter_option = st.selectbox(
            "Select emails to view:",
            ["All", "Spam", "Not Spam"]
        )

        if filter_option == "Spam":
            filtered_df = df[df["prediction"] == "spam"]
        elif filter_option == "Not Spam":
            filtered_df = df[df["prediction"] == "ham"]
        else:
            filtered_df = df

        st.dataframe(
            filtered_df[["email_id", "subject", "prediction"]],
            use_container_width=True
        )

        st.subheader("✉️ Test a New Email")

        user_subject = st.text_input("Email Subject")
        user_body = st.text_area("Email Body")

        if st.button("Check Spam"):
            user_text = user_subject + " " + user_body
            result = model.predict([user_text])[0]

            if result == "spam":
                st.error("🚨 This email is SPAM")
            else:
                st.success("✅ This email is NOT spam")