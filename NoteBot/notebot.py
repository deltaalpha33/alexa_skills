from flask import Flask
from flask_ask import Ask, statement, question, session
import requests
import time
#import unidecode
import json
import numpy as np
from face_label import face_labeling_personalized as fl
from audio_input import audio
from note_code import notepy
"""
Basic format for the skill:


Start:
    Recognized:
        Leave Note:
            Alexa asks for a name
            record and exit|
        Read Notes:
            read all notes and exit|
    More than 1:
        Alexa says to check if you are exclusively visible and exits|
    Not Recognized:
        Alexa asks for your name
        State a name:
            take picture and quit|
        Cancel:
            quit|

"""



print(type(-1))

note = notepy.Note("notes",audio)
note.loadDBnp()
DB = fl.loadDBnp("vectors")



app = Flask(__name__)
ask = Ask(app, '/')

#states:
#0: Begin
#1: Asked for Name
#2: asked for leave/read
#3: Set to record
#4: Set to read.

state = 0
name = ""
nameGoal = ""
@app.route('/')
def homepage():
    return "Hello"


@ask.launch
def start_skill():
    return question("Tell me to start.")

@ask.intent("StartIntent")
def start_skill():
    global name
    global state
    global nameGoal
    state = 0
    name = fl.findMatch(fl.get_desc(fl.take_picture()),DB)
    if name == fl.tooManyString:
        return statement("Make sure you are visible and no one else is.")
    if name == fl.noMatchString:
        state  = 1
        return question("I don't know you. What is your name?")

    state  = 2
    msg = "Hi " + name + ", would you like to read or leave a note?"
    return question(msg)

@ask.intent("LeaveNoteIntent", default={'forName': 'holder'})
def leave_note(forName):
    global name
    global state
    global nameGoal
    if state  == 2:
        #state  = 3
        nameGoal = forName
        print("starting...")
        print(note.db)
        note.add_note(30,nameGoal)
        print(note.db)
        print("ended.")
        state = 2
        return statement(forName + " what?")
    return statement("That didn't make sense.")

@ask.intent("ReadNoteIntent")
def read_note():
    global name
    global state
    global nameGoal
    if state  == 2:
        #state  = 4
        state = 2
        print("Reading...")
        print(note.db)
        note.loadDBnp()
        note.read_notes(name.lower())
        print(note.db)
        print("ended.")
        return statement("")
    return statement("That didn't make any sense.")
"""
@ask.intent("AddFaceIntent")
def add_face(forName):
    global name
    global state
    global nameGoal
    if state == 1:
        addFaceDB(forName,)
        return statement("You are now in my database.")
    return statement("That didn't make any sense.[]")
"""

@ask.intent("AddFaceIntent")
def add_face(forName):
    global name
    global state
    global nameGoal
    if state == 1:
        state = 2
        result = fl.addImgToDB(DB,fl.take_picture(),forName,"vectors")
        return statement(result)
    return statement("That didn't make any sense.")

@ask.intent("CancelIntent")
def cancel():
    global name
    global state
    global nameGoal
    if state == 1:
        state = 2
        return statement("Smell ya later.")
    return statement("That didn't make any sense.")


@ask.session_ended
def session_ended():
    global name
    global state
    global nameGoal
    print ("success")
    if state  == 3:
        #note.add_note(30,nameGoal)
        print("suc my cess")
    return "{}",200

if __name__ == '__main__':
    app.run(debug=True)