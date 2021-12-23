import nltk
import pyttsx3
import pickle
import numpy as np
import json
import random
import arabic_reshaper
import os
from nltk.stem import WordNetLemmatizer
from gtts import gTTS
from googletrans import Translator
from keras.models import load_model
from tkinter import *
from googletrans import Translator

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
translator = Translator()
engine = pyttsx3.init()
# Language in which you want to convert
language = 'en'
from flask import Flask,jsonify

app = Flask(__name__)


@app.route('/bot',methods=['POST'])
def response():
  def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

  from googletrans import Translator
  from googletrans.gtoken import TokenAcquirer
  translator = Translator()
#flutter-chatbotapp54654
  def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_rec = translator.translate(sentence, dest='ar')
    reshaped_text = arabic_reshaper.reshape(sentence_rec.text)
    if (sentence_rec == 'ar'):
      arabic_reshaper.reshape(sentence_rec.text)
      rec_text = reshaped_text[::-1]
      rec_text = translator.translate(sentence, dest='en')
      sentence_words = clean_up_sentence(rec_text)
      print(rec_text)
      print(rec_text)
    else:
      sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
      for i, w in enumerate(words):
        if w == s:
          # assign 1 if current word is in the vocabulary position
          bag[i] = 1
          if show_details:
            print("found in bag: %s" % w)
    return (np.array(bag))

  def predict_class(sentence, model):  # ******
    # filter out predictions below a threshold

    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
      return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

  def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
      if (i['tag'] == tag):
        result = random.choice(i['responses'])
        break
    return result

  def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

  # Creating GUI with tkinter
  import tkinter
  from googletrans import Translator
  import arabic_reshaper
  def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    received = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
      # make an object from translator
      translator = Translator()
      #
      ChatLog.config(state=NORMAL)

      receive = translator.translate(received, dest='ar')
      reshaped_text = arabic_reshaper.reshape(receive.text)
      if (receive.src == 'ar'):
        arabic_reshaper.reshape(receive.text)
        rec_text = reshaped_text[::-1]
        ChatLog.insert(END, "You: " + rec_text + '\n\n')
      else:

        ChatLog.insert(END, "You: " + msg + '\n\n')
      # make the object from translator detect the source of the enyered language
      FromUser = translator.translate(msg).src
      #
      ChatLog.config(foreground="#442265", font=("Verdana", 12))

      res = chatbot_response(msg)
      # make the destination language same as the source
      result = translator.translate(res, dest=FromUser)
      reshaped_text = arabic_reshaper.reshape(result.text)
      if (result.dest == 'ar'):
        arabic_reshaper.reshape(result.text)
        rev_text = reshaped_text[::-1]
        ChatLog.insert(END, "Bot: " + rev_text + '\n\n')
        # myobj = gTTS(text=result.text, lang='fr', slow=False)
        # myobj.save("welcome.mp3")
        # Playing the converted file
        # os.system("welcome.mp3")
      else:
        ChatLog.insert(END, "Bot: " + result.text + '\n\n')
        say = result.text
        converter = pyttsx3.init()
        converter.setProperty('rate', 85)
        voice_id_Male = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0"
        voice_id_female = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"

        # Use female voice
        converter.setProperty('voice', voice_id_female)
        engine.say(say)
        engine.runAndWait()

if __name__=="__main__":
    app.run(host="0.0.0.0",)
