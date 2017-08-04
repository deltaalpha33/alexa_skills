# Song FP

An Alexa skill for identifying songs.

`Song FP` was created as a prototype for the CogWorks 2017 summer program, in the [Beaver Works Summer Institute at MIT](https://beaverworks.ll.mit.edu/CMS/bw/bwsi). It was developed by [Daschel Cooper](https://github.com/thedashdude).

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
* `matplotlib`
* `scipy`

To load your songs to pickle files of fingerprints, open `song_labeling_notebook.ipynp` and follow the instructions to load mp3s. 

To set up the server run `songfp_skill.py`

```shell
python songfp_skill.py
```

Use ngrok to tunnel port 5000:

```shell
ngrok http 5000
```


## Alexa Setup

First link your amazon developer account to your Alexa.

Then go to the [Alexa Skills Kit](https://developer.amazon.com/edw/home.html#/skills) and create the Song FP skill.

Under configuration enter the adress ngrok generated. It looks like `https://XXXXXXXX.ngrok.io`.

Under interation model enter the intent schema and sample utterances found in `skill_setup.txt`.

## Use

The basic format for the skill is as follows:

Tell Alexa:

- ""Alexa, ask song f.p. what song is this?"
- ""Alexa, ask song f.p. to identify?"