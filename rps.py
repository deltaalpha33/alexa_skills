from flask import Flask
from flask_ask import Ask, statement, question
from roshambot import *
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
beat={'rock':'paper', 'paper':'scissors', 'scissors':'rock'}
score = [0, 0]
move_list = ['', '']
#memory is the number of moves to keep track of in computing probabilities.
#Our data requirements for high confidence are exponential in this variable,
#but should eventually get better predictions with higher memory
memory=2
last=['r', 'p', 's']
mtuples=last[:]
#Generate all rps tuples of length memory
for i in range(memory-1):
    new=[]
    for a in last: new.append( 'r' + a )
    for a in last: new.append( 'p' + a )
    for a in last: new.append( 's' + a )
    last=new
mtuples=last

#Observed mtuples of opponent moves, initialized at one observation each.
data={ a: [1,1,1] for a in mtuples }
@ask.launch
def start_skill():
    msg = "Hello. Would you like to play rock, paper, scissors?"
    return question(msg)

@ask.intent("ScoreIntent")
def outputscore():
    return question("You have {} and I have {}".format(score[0], score[1]))

def randommove():
    m = np.random.choice(['rock', 'paper', 'scissors'])
    return m

def compute_probabilities():
    #Using Bayes' theorem, we get that:
    #  P(H|D) = #(observations of D and H) / #(observations of D)
    if len(move_list[0])<memory:
        return [1.0/3 for i in range(3)]

    D = move_list[0][-memory:]
    N = sum(data[D])
    P = [ 1.0*x/N for x in data[D] ]
    print(P)
    return P

def trump(m):
    return beat[m]

def get_move():
    if len(move_list[0])<memory: return random.choice(['rock', 'paper', 'scissors'])
    P=compute_probabilities()
    #choose a random guess using probabilities.
    x=random.random()
    if x<P[0]: return trump( 'rock' )
    if x<P[0]+P[1]: return trump( 'paper' )
    return trump( 'scissors' )

@ask.intent("MoveIntent")
def play(move):
    d = {}
    d['rock'] = 0
    d['paper'] = 1
    d['scissors'] = 2
    m = get_move()
    #msg = "Ok, thanks. Have a nice day."
    move_list[0] += move[0]
    move_list[1] += m[0]
    #Update data tables.
    if len(move_list[0])>memory:
        #m is the previous move
        new_data = move_list[0][-memory-1:-1]
        j='rps'.index( move[0] )
        data[new_data][j]+=1
    if move == m:
        return question("{}! We tied! Again?".format(m))
    playerwin = 2 - (d[move] - d[m]) % 3
    if playerwin:
        score[0] += 1
        return question("{}, You win! Again?".format(m))
    else:
        score[1] += 1
        return question("{}, I win! Again?".format(m))

@ask.intent("YesIntent")
def startgame():
    msg = "What's your move?"
    return question(msg)

@ask.intent("NoIntent")
def no_intent():
    msg = "Ok, thanks. Have a nice day."
    return statement(msg)

if __name__ == '__main__':
    app.run(debug=True)
