import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model("modelET.h5")

# Load the CSV files
train_df = pd.read_csv("training.csv")
test_df = pd.read_csv("test.csv")
validation_df = pd.read_csv("validation.csv")

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text'])

# Streamlit app
def main():
    st.title("Emotion Detection from Text")

    # Input text area for user input
    text_input = st.text_area("Enter text:", "")

    # Button to trigger emotion detection
    if st.button("Detect Emotion"):
        # Perform emotion detection
        emotion = predict_emotion(text_input)
        st.write("Predicted Emotion:", emotion)

# Function to preprocess text and perform emotion detection
def predict_emotion(text):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')

    # Perform emotion detection
    prediction = model.predict(padded_sequence)

    # Convert prediction to emotion label
    emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[prediction.argmax()]
    
    return predicted_emotion

if __name__ == "__main__":
    main()
