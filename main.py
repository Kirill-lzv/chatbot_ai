import os
from background import keep_alive
import telebot
from telebot import types

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



TOKEN = os.environ['TOKEN']
bot = telebot.TeleBot(TOKEN)



questions = []
answers = []

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
total_words = len(tokenizer.word_index) + 1



input_sequences = tokenizer.texts_to_sequences(questions)
input_sequences = pad_sequences(input_sequences)


answer_labels = keras.utils.to_categorical(np.arange(len(answers)), num_classes=len(answers))

model = tf.keras.models.load_model("model_700.keras")
def generate_response(user_input):
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_seq = pad_sequences(input_seq, maxlen=input_sequences.shape[1])
    predicted_idx = np.argmax(model.predict(input_seq))
    return answers[predicted_idx]


@bot.message_handler(commands=['start'])
def start(message):
  bot.send_message(message.chat.id, '', reply_markup=keyboard())


@bot.message_handler(content_types=['text'])
def send_text(message):
  if message.text == '🧰Cайт':
    bot.send_message(message.chat.id, '')
  elif message.text == '🚨Тех.поддержка':
    bot.send_message(message.chat.id, '')
  elif message.text == '🔴Telegram канал':
    bot.send_message(message.chat.id, '')
  else:
    response = generate_response(message.text)
    bot.send_message(message.chat.id, response)
  
  
def keyboard():
  markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
  btn2 = types.KeyboardButton('🧰Cайт')
  btn3 = types.KeyboardButton('🚨Тех.поддержка')
  btn4 = types.KeyboardButton('🔴Telegram канал')
  
  markup.add(btn2, btn3, btn4)
  return markup



bot.infinity_polling(none_stop=True)