from flask import Flask
from flask_ask import Ask, statement, question
import requests
import time
#import unidecode
import json
import numpy as np
from face_label import face_labeling_personalized as fl
from audio_input import audio

print(type(-1))
a = audio.Audio()
DB = fl.loadDBnp("vectors")


app = Flask(__name__)
ask = Ask(app, '/')

#0: Begin
#1: Asked for Name
#2: asked for leave/read
#
#
#
#

state = 0

@app.route('/')
def homepage():
    return "Hello"


@ask.launch
def start_skill():
    name = fl.findMatch(fl.get_desc(fl.take_picture()),DB)
    if name == fl.tooManyString:
        return statement("There are too many people. Make sure You are the only person visible.")
    if name == fl.noMatchString:
        state = 1
        return question("I don't know you. What is your name?")

    state = 2
    msg = "Hi " + name + ", would you like to read or leave a note?"
    return question(msg)


@ask.intent("YesIntent")
def identify():
    name = fl.findMatch(fl.get_desc(fl.take_picture()),DB)
    if name == "I don't know":
        return statement("I don't ")
    return statement(saythis)



@ask.intent("NoIntent")
def close():
    msg = "Ok. Have a nice day."
    return statement(msg)

if __name__ == '__main__':
    app.run(debug=True)