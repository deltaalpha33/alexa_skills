from flask import Flask
from flask_ask import Ask, statement, question
import requests
import time
#import unidecode
import json
import numpy as np
from face_label import face_labeling as fl

DB = fl.loadDBnp("vectors")



app = Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Hello"

@ask.launch
def start_skill():
    msg = "Hello. Would you like me to identify who you are?"
    return question(msg)

@ask.intent("YesIntent")
def listen():
    saythis = fl.label_faces_text( fl.take_picture(),DB )

    return statement(saythis)

@ask.intent("NoIntent")
def no_intent():
    msg = "Ok. Have a nice day."
    return statement(msg)

if __name__ == '__main__':
    app.run(debug=True, port=4000)