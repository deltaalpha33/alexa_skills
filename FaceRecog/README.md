# Race Rec

An Alexa skill for naming people in front of a camera.

`Face Rec` was created as a prototype for the CogWorks 2017 summer program, in the [Beaver Works Summer Institute at MIT](https://beaverworks.ll.mit.edu/CMS/bw/bwsi). It was developed by [Daschel Cooper](https://github.com/thedashdude).

## Running Instructions

Install all necessary programs and python packages:
##### Programs:
* [ngrok](https://ngrok.com/)

##### Packages:
* `numpy`
* `dlib_models`
* `matplotlib`
* `Flask`
* `flask_ask`

To set up your own database of face descriptors, run `Face-Recognition.ipynb` and follow the instructions.

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

Then go to the [Alexa Skills Kit](https://developer.amazon.com/edw/home.html#/skills) and create the Face Rec skill.

Under configuration enter the adress ngrok generated. It looks like `https://XXXXXXXX.ngrok.io`.

Under interation model enter the intent schema and sample utterances found in `skill_setup.txt`.

## Use

The basic format for the skill is as follows:

- "Alexa, ask face rec who we are."