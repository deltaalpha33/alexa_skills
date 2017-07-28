from flask import Flask
from flask_ask import Ask, statement, question
import requests
import time
import unidecode
import json
import numpy as np

app = Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Hello"

score = [0, 0]
@ask.launch
def start_skill():
    msg = "Hello. Would you like to play rock, paper, scissors?"
    return question(msg)

@ask.intent("ScoreIntent")
def outputscore():
    return statement("You have {} and I have {}".format(score[0], score[1]))
@ask.intent("MoveIntent")
def play(move):
    d = {}
    d['rock'] = 0
    d['paper'] = 1
    d['scissors'] = 2
    m = np.random.choice(['rock', 'paper', 'scissors'])
    #msg = "Ok, thanks. Have a nice day."
    playerwin = d[m] < d[move] or d[move] - d[m] == -2
    if move == m:
        return statement("{}! We tied!".format(m))
    if playerwin:
        score[0] += 1
        return statement("{}, You win!".format(m))
    else:
        score[1] += 1
        return statement("{}, I win!".format(m))

@ask.intent("YesIntent")
def startgame():
    msg = "What's your move?"
    return statement(msg)

@ask.intent("NoIntent")
def no_intent():
    msg = "Ok, thanks. Have a nice day."
    return statement(msg)

if __name__ == '__main__':
    app.run(debug=True)