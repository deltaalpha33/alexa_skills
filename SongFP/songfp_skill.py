from flask import Flask
from flask_ask import Ask, statement, question
import requests
import time
#import unidecode
import json
import numpy as np
from audio_input import audio
from song_label import song_labeling as sl

fp = sl.FingerPrinter()
a = audio.Audio()
fp.loadDB("songsDB.p","namesDB.p")



app = Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Hello"

@ask.launch
def start_skill():
    msg = "Hello. Would you like me to identify this song?"
    return question(msg)

@ask.intent("YesIntent")
def listen():
    samples = a.read_mic(7)
    name = fp.best(fp.match_song(fp.get_finger_print(fp.get_peaks(samples))))[:-4]

    return statement(name)

@ask.intent("NoIntent")
def no_intent():
    msg = "Ok. Have a nice day."
    return statement(msg)

if __name__ == '__main__':
    app.run(debug=True)