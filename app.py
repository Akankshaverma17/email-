import streamlit as st
from transformers import pipeline
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import re

st.set_page_config(page_title="AI vs Real Image Detection", layout="wide")

st.title("🧠 AI vs Real Image Detection with Email")

# -----------------------------
# Load AI Model
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="umm-maybe/AI-image-detector"
    )

detector = load_model()

# -----------------------------
# Email Validation
# -----------------------------
def is_valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email)

# -----------------------------
# Send Email Function
# -----------------------------
def send_email(sender_email, receiver_email, result_label, confidence):

    sender_password = st.secrets["APP_PASSWORD"]

    subject = "AI Image Detection Result"
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = f"""
AI Image Detection Result

Result: {result_label}
Confidence: {confidence*100:.2f}%
Time: {time_now}
"""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

# -----------------------------
# UI SECTION (Always Visible)
# -----------------------------
st.subheader("📧 Email Settings")

sender_email = st.text_input("Sender Gmail Address")
receiver_email = st.text_input("Receiver Email Address")

st.subheader("🖼 Upload Image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -----------------------------
# Prediction Section
# -----------------------------
prediction_done = False

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    result = detector(image)
    label = result[0]["label"]
    confidence = result[0]["score"]

    st.subheader("🔍 Prediction Result")
    st.write(f"Result: {label}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    prediction_done = True

# -----------------------------
# Send Button (Always Visible)
# -----------------------------
if st.button("📨 Send Result via Email"):

    if not prediction_done:
        st.error("Please upload and predict an image first.")
    elif not sender_email or not receiver_email:
        st.error("Please enter both email addresses.")
    elif not is_valid_email(sender_email) or not is_valid_email(receiver_email):
        st.error("Enter valid email addresses.")
    else:
        try:
            send_email(sender_email, receiver_email, label, confidence)
            st.success("✅ Email sent successfully!")
        except Exception as e:
            st.error(f"Error sending email: {e}")