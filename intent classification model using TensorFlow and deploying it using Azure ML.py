import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# Step 1: Prepare the dataset
data = {
    "greetings": ["hello", "hi", "hey", "good morning", "good evening"],
    "farewell": ["bye", "goodbye", "see you later", "take care"],
    "thanks": ["thank you", "thanks", "much appreciated", "grateful"],
}

sentences = []
labels = []
label_map = {}

# Map intent labels to integers and create sentences
for idx, (intent, phrases) in enumerate(data.items()):
    label_map[idx] = intent
    for phrase in phrases:
        sentences.append(phrase)
        labels.append(idx)

# Step 2: Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1  # +1 because index starts from 1
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')
labels = np.array(labels)

# Step 3: Build the model
model = Sequential([
    Embedding(vocab_size, 16, input_length=padded_sequences.shape[1]),  # Embedding layer
    SpatialDropout1D(0.2),  # Dropout layer to prevent overfitting
    LSTM(16, dropout=0.2, recurrent_dropout=0.2),  # LSTM layer for sequence processing
    Dense(16, activation='relu'),  # Dense layer with ReLU activation
    Dense(len(data), activation='softmax')  # Output layer (softmax for classification)
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the model
model.fit(padded_sequences, labels, epochs=50, verbose=1)

# Step 5: Save the model
model.save("intent_model.h5")

# Step 6: Function for predicting intent
def predict_intent(text):
    # Tokenize and pad the input text in the same way as the training data
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=padded_sequences.shape[1], padding='post')
    
    # Make prediction
    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Get the intent label from the prediction
    intent = label_map[predicted_class]
    return intent

# Example usage of the trained model for prediction
text_input = "hello"
predicted_intent = predict_intent(text_input)
print(f"The predicted intent for '{text_input}' is: {predicted_intent}")
