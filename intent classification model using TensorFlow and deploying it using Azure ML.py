import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget, AmlCompute

# Step 1: Prepare the dataset
data = {
    "greetings": ["hello", "hi", "hey", "good morning", "good evening"],
    "farewell": ["bye", "goodbye", "see you later", "take care"],
    "thanks": ["thank you", "thanks", "much appreciated", "grateful"],
}

sentences = []
labels = []
label_map = {}

for idx, (intent, phrases) in enumerate(data.items()):
    label_map[idx] = intent
    for phrase in phrases:
        sentences.append(phrase)
        labels.append(idx)

# Step 2: Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')
labels = np.array(labels)

# Step 3: Build the model
model = Sequential([
    Embedding(vocab_size, 16, input_length=padded_sequences.shape[1]),
    SpatialDropout1D(0.2),
    LSTM(16, dropout=0.2, recurrent_dropout=0.2),
    Dense(16, activation='relu'),
    Dense(len(data), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the model
model.fit(padded_sequences, labels, epochs=50, verbose=1)

# Step 5: Save the model
model.save("intent_model.h5")

# Step 6: Set up Azure ML
ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name="intent-classification")

# Step 7: Define the environment
env = Environment.from_conda_specification(name='tensorflow-env', file_path='environment.yml')

# Step 8: Define compute target
compute_target = ComputeTarget.create(ws, 'cpu-cluster', AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2'))
compute_target.wait_for_completion(show_output=True)

# Step 9: Submit training script
src = ScriptRunConfig(source_directory=".", script="train.py", compute_target=compute_target, environment=env)
run = experiment.submit(src)
run.wait_for_completion(show_output=True)

# Step 10: Register the model
model = run.register_model(model_name='intent_model', model_path='outputs/intent_model.h5')
