# Note Bot

An Alexa skill for leaving and recieving notes through facial recognition.

`Note Bot` was created as a prototype for the CogWorks 2017 summer program, in the [Beaver Works Summer Institute at MIT](https://beaverworks.ll.mit.edu/CMS/bw/bwsi). It was developed by [Daschel Cooper](https://github.com/thedashdude).

## Running Instructions

Install all necessary programs and python packages:
##### Programs:
* [ngrok](https://ngrok.com/)

##### Packages:
* `numpy`
* `dlib_models`
* `librosa`
* `pyaudio`
* `Flask`
* `flask_ask`

To set up the server run `facerec_skill.py`

```shell
python facerec_skill.py
```

Use ngrok to tunnel port 5000:

```shell
ngrok http 5000
```

## Alexa Setup

First link your amazon developer account to your Alexa.

Then go to the [Alexa Skills Kit](https://developer.amazon.com/edw/home.html#/skills) and create the Note Bot skill.

Under configuration enter the adress ngrok generated. It looks like `https://XXXXXXXX.ngrok.io`.

Under interation model enter the intent schema and sample utterances found in `skill_setup.txt`.

If you expect any uncommon names, you can add them to AMAZON.US_FIRST_NAME in the custom slot types area.

## Use

The basic format for the skill is as follows:

Begin by telling alexa to start note bot

- "Alexa, ask note bot to start"
- "Alexa, ask note bot to begin"

If she recognizes you you can either read your notes,

- "read my notes"
- "play"
- "read"

or leave a note

- "leave a note for victor"
- "leave lucia a note"

If she doesn't:

Either say your name

- "I'm daschel"
- "my name is daschel"
- "call me daschel"

or cancel the program

- "cancel"
- "stop"