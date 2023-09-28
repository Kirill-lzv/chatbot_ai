import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

questions = []
answers = []

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
total_words = len(tokenizer.word_index) + 1

input_sequences = tokenizer.texts_to_sequences(questions)
input_sequences = pad_sequences(input_sequences)

model = keras.Sequential([
    keras.layers.Embedding(total_words, 100, input_length=input_sequences.shape[1]),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(answers), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

answer_labels = keras.utils.to_categorical(np.arange(len(answers)), num_classes=len(answers))

model.fit(input_sequences, answer_labels, epochs=4000)

model.save('model_1.keras')


def generate_response(user_input):
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_seq = pad_sequences(input_seq, maxlen=input_sequences.shape[1])
    predicted_idx = np.argmax(model.predict(input_seq))
    return answers[predicted_idx]
