import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load datasets
train_df = pd.read_csv('/Users/shreya/Desktop/capstone/Emotion_text/training.csv')
test_df = pd.read_csv('/Users/shreya/Desktop/capstone/Emotion_text/test.csv')
validation_df = pd.read_csv('/Users/shreya/Desktop/capstone/Emotion_text/validation.csv')

# Load the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text'])

# Define maximum sequence length
maxlen = 100

# Load the model
model = load_model('modelET.h5')

# Function to preprocess text
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='post')
    return padded_sequence

# Define emotion labels
emotion_labels = {0: 'Anger', 1: 'Fear', 2: 'Joy', 3: 'Sadness', 4: 'Surprise'}

# Main function to run the app
def main():
    st.title("Emotion Detection from Text")

    # Input text area for user input
    text_input = st.text_area("Enter text:", "")

    # Button to trigger emotion detection
    if st.button("Detect Emotion"):
        # Preprocess the input text
        processed_text = preprocess_text(text_input)
        
        # Perform inference
        prediction = model.predict(processed_text)
        
        # Convert prediction to emotion label
        predicted_emotion = emotion_labels[prediction.argmax()]
        
        # Display the predicted emotion
        st.write("Predicted Emotion:", predicted_emotion)

# Run the main function
if __name__ == "__main__":
    main()
