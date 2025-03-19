import streamlit as st
import speech_recognition as sr
import pandas as pd
import os
import sounddevice as sd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download required NLTK models (only once)
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# CSV file to store expenses
CSV_FILE = "expenses.csv"

# Initialize recognizer
recognizer = sr.Recognizer()

def recognize_speech():
    samplerate = 44100  # Standard audio rate
    duration = 5  # Record for 5 seconds
    st.info("Listening... Speak now!")
    
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    
    audio_data = sr.AudioData(audio.tobytes(), samplerate, 2)
    
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition service error"

def extract_expense(text):
    words = word_tokenize(text.lower())  # Tokenize text
    tagged_words = pos_tag(words)  # POS tagging

    amount = None
    category = "Unknown"

    for word, tag in tagged_words:
        if tag == "CD":  # CD = Cardinal Number (used for money amounts)
            try:
                amount = float(word.replace("₹", ""))
            except ValueError:
                pass
        elif tag in ["NN", "NNS", "JJ"]:  # Use Nouns/Adjectives as categories
            category = word  

    return amount, category

def save_expense(amount, category):
    if amount is None:
        return "Could not detect amount"

    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["Date", "Amount", "Category"])

    new_entry = pd.DataFrame({
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Amount": [amount],
        "Category": [category]
    })
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    return f"Saved: ₹{amount} for {category}"

# Streamlit UI
st.title("🎤 Voice-Based Expense Tracker (₹)")

if st.button("Record Expense"):
    text = recognize_speech()
    st.write(f"Recognized: {text}")
    amount, category = extract_expense(text)
    result = save_expense(amount, category)
    st.success(result)

# Display existing expenses
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    st.subheader("📊 Expense History")
    st.dataframe(df)

    # Chart options
    st.subheader("📈 Expense Visualization")
    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart", "Line Chart"])

    # Dynamic category selection
    unique_categories = df["Category"].unique().tolist()
    selected_categories = st.multiselect("Select Categories to Display", unique_categories, default=unique_categories)

    # Filter data based on selected categories
    df_filtered = df[df["Category"].isin(selected_categories)]

    if not df_filtered.empty:
        if chart_type == "Bar Chart":
            category_totals = df_filtered.groupby("Category")["Amount"].sum()
            st.bar_chart(category_totals)

        elif chart_type == "Pie Chart":
            category_totals = df_filtered.groupby("Category")["Amount"].sum()
            fig, ax = plt.subplots()
            ax.pie(category_totals, labels=category_totals.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")  
            st.pyplot(fig)

        elif chart_type == "Line Chart":
            df_filtered["Date"] = pd.to_datetime(df_filtered["Date"])  
            df_pivot = df_filtered.pivot(index="Date", columns="Category", values="Amount").fillna(0)
            st.line_chart(df_pivot)
