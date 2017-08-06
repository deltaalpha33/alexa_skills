## IMPORTANT

to run properly the absolute path of the sanitized poetry dataset must be specified in the initialization of Poet in Poet.py

## load n gram model from disk - the location of the model is in the sanitized poetry dataset and must be specified in the initialization function of Poet
Poet.load_model_ngram()
Poet.nomalize_model()


## to generate poetery

Poet.generate_text()