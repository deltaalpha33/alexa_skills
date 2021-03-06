from flask import Flask
from flask_ask import Ask, statement, question
import requests
import time
import unidecode
import json
from poetry_gen import poet
app = Flask(__name__)
ask = Ask(app, '/')

#import io
# with io.open('/home/delta/mit-course/git/alexa_skills/poetry_dataset/35SonnetsbyFernandoAntnioNogueiraPessoa19978.txt' ,'r',encoding='utf8') as f:
#     text = f.read()

#     test_poem = text.encode('ascii', 'ignore')
#     test_poem = str(test_poem)
#     print(test_poem)


alexa_poet = poet.Poet()
alexa_poet.load_model_ngram()
alexa_poet.nomalize_model()
@app.route('/')
def homepage():
    return "Hello"

@ask.launch
def start_skill():
    msg = "Would you like to listen to some poetry"
    return question(msg)

@ask.intent("YesIntent")
def make_poetry():
    return statement(alexa_poet.generate_text()[10:350])

@ask.intent("NoIntent")
def no_intent():
    msg = "Have a nice day."
    return statement(msg)

if __name__ == '__main__':
    app.run(debug=True)


#alexa config 

# {
#   "intents": [
#     {
#       "intent": "YesIntent"
#     },
#     {
#       "intent": "NoIntent"
#     }
#   ]
# }

# YesIntent yes
# NoIntent No