import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import re

st.set_page_config(page_title="AI vs Real Image Detection", layout="wide")

st.title("🧠 AI vs Real Image Detection")
st.write("Upload an image and get the result via Email.")

# -----------------------------
# Load AI Detection Model
# -----------------------------
@st.cache_resource
def load_model():
    detector = pipeline(
        "image-classification",
        model="umm-maybe/AI-image-detector"
    )
    return detector

detector = load_model()

# -----------------------------
# Email Validation Function
# -----------------------------
def is_valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email)

# -----------------------------
# Email Sending Function
# -----------------------------
def send_email_result(receiver_email, result_label, confidence):

    sender_email = st.secrets["EMAIL"]
    sender_password = st.secrets["APP_PASSWORD"]

    subject = "AI Image Detection Result"
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = f"""
AI Image Detection Result

Result: {result_label}
Confidence: {confidence*100:.2f}%
Time: {time_now}

Thank you for using the AI Detection App.
"""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email Error: {e}")
        return False

# -----------------------------
# Session History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Email Input Field (Dynamic Receiver)
receiver_email = st.text_input("📧 Enter Email to Receive Result")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and receiver_email:

    if not is_valid_email(receiver_email):
        st.error("❌ Please enter a valid email address.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        result = detector(image)
        label = result[0]["label"]
        confidence = result[0]["score"]

        st.subheader("Prediction Result")
        st.write(f"**Result:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        # Send Email
        if send_email_result(receiver_email, label, confidence):
            st.success("📧 Result sent successfully!")

        # Save history
        st.session_state.history.append({
            "Filename": uploaded_file.name,
            "Prediction": label,
            "Confidence (%)": round(confidence*100, 2),
            "Sent To": receiver_email
        })

# -----------------------------
# Filter Section
# -----------------------------
if st.session_state.history:
    st.subheader("📊 Prediction History")

    df = pd.DataFrame(st.session_state.history)

    filter_option = st.selectbox(
        "Filter Results",
        ["All"] + list(df["Prediction"].unique())
    )

    if filter_option != "All":
        df = df[df["Prediction"] == filter_option]

    st.dataframe(df, use_container_width=True)