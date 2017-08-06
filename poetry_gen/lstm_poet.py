import numpy as np

from collections import Counter

import os
import io
import glob

import re, string
import pickle
import codecs

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


import itertools

from keras.models import load_model



class Poet():
    def __init__(self):

        """
        Decode text file from bytes to string, then make lowercase for processing.
        Then, initialize variables.

        Parameters
        ----------
        path : str
            File path to text, ex: poems.txt
        
        """
        self.documents = list()
        self.bag_of_words = Counter()
        self.sanitized_poetry_dataset_dir = "/home/delta/mit-course/git/alexa_skills/sanitized_poetry_dataset/"


        self.token_id_dict = dict()

        self.terminator_token = 0

    def sanitize_text(self, path):

        """
        Decode text file from bytes to string, then make lowercase for processing.

        Parameters
        ----------
        path : str
            File path to text, ex: poems.txt

        """

        file_paths = glob.glob(path +'*.txt')
        punc_regex_string = [ "(" + re.escape(escape_char) + ")" + "|" for escape_char in string.punctuation.replace("'", '').replace("\\", '')]
        #punc_regex = re.compile('([{}])*'.format(re.escape(string.punctuation.replace("'", '').replace("\\", ''))))

        punc_regex = re.compile(''.join(punc_regex_string))
        alpha_split = re.compile("([A-Z][^A-Z]*)*") # remove all-caps words
        number_regex = re.compile('[[0-9]+]|[IVXLCM]+') # remove numbers & Roman Numerals

        #print(re.sub(number_regex, "", ("abc IV")))

        for file_num, file_path in enumerate(file_paths):
            
            with io.open(file_path,'r',encoding='utf8') as f:
                unicode_data = f.read()

                document = str(unicode_data.encode('ascii', 'ignore')).replace("\\", '').replace(" '", "'")
                document = str(unicode_data.encode('ascii', 'ignore')).replace("\\", '')

                token_list = document.split()
                
                token_list =    [re.sub(number_regex, "", token) for split_punc in (punc_regex.split(split_punc_item)
                                 for split_punc_item in (item for small_list in (alpha_split.split(word)

                                 for word in token_list) for item in small_list if item != '' and item != " ") ) for token in split_punc if token != None and token != '']

                # token_list = list((item for small_list in (alpha_split.split(word)

                #                  for word in document.split()) for item in small_list if item != '' and item != " ") )
                
                #print(token_list)

                with open(self.sanitized_poetry_dataset_dir + str(file_num) + ".txt", 'w') as f:
                    f.write(' '.join(token_list))

    def load_token_dict(self):
        with open(self.sanitized_poetry_dataset_dir + "token_dict.pkl", 'rb') as f:
            self.token_id_dict = pickle.load(f)

    def create_token_dict(self):

        self.token_id_dict = { token : token_id for token_id , token in enumerate(self.get_unique_tokens(), start=1)}
        total_values = len(self.token_id_dict) + 1
        #self.token_id_dict = { normal_token : token_id / total_values for normal_token , token_id in self.token_id_dict.items()}


        with open(self.sanitized_poetry_dataset_dir + "token_dict.pkl", 'wb') as f:
            pickle.dump(self.token_id_dict, f, pickle.HIGHEST_PROTOCOL)

    def document_generator(self):
        #space_regex = re.compile(' +')
        file_paths = glob.glob(self.sanitized_poetry_dataset_dir +'*.txt')
        for file_path in file_paths:

            with open(file_path, 'r') as f:
                document = f.read()
                #document = space_regex.sub(' ', document)
                #document = document.split(" ")[1:]
                yield document

    def get_unique_tokens(self):
        return list({char for document in self.document_generator() for char in document})


    def get_tokenized_string(self, window_size):
        for document in self.document_generator():

                left_over_chars = len(document) % window_size
                padding = window_size - left_over_chars

                for cnt in range(len(document) - window_size - left_over_chars):
                    yield (self.token_id_dict[token] for token in document[cnt: cnt + window_size])

                yield (self.token_id_dict[document[cnt]] if  cnt < left_over_chars else self.terminator_token for cnt in range(left_over_chars + 1))
        return None

    def data_generator(self, data_dim, timesteps, num_data_points = 1000):
        token_generator = self.get_tokenized_string(timesteps)
        previous_data_points = list()
        data_points = list()

        for cnt, token_itr in enumerate(token_generator, start = 1):
            #print(list(token_itr))
            data_point = np.fromiter(token_itr, float)
            data_point = np_utils.to_categorical(data_point, num_classes=data_dim)
            #print(data_point.shape)
            data_points.append(data_point)
            if cnt % num_data_points == 0:
                previous_data_points = np.array(data_points)
                data_points = list()
                break

        for cnt, token_itr in enumerate(token_generator, start = 1):
            data_point = np.fromiter(token_itr, float)
            data_point = np_utils.to_categorical(data_point, num_classes=data_dim)
            #print(data_point.shape)
            data_points.append(data_point)
            if cnt % num_data_points == 0:
                data_points = np.array(data_points)
                yield (previous_data_points , data_points)
                previous_data_points = data_points
                data_points = list()




    def train_lstm(self):

        alexa_poet.load_token_dict()

        # as the first layer in a Sequential model

        # string_size = 100
        # num_strings = 1000

        # train_data = itertools.islice(self.get_tokenized_string(string_size), num_strings) # grab the first five elements
        # X = np.array(list(train_data))


        data_dim = len(self.token_id_dict) + 1 #terminator not included in token dictionary
        timesteps = 8
        num_classes = data_dim # it is learning to predict the data
        batch_size = 200

        # expected input data shape: (batch_size, timesteps, data_dim)
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, stateful=True,
                       batch_input_shape=(batch_size, timesteps, data_dim)))
        model.add(LSTM(32, return_sequences=True, stateful=True))
        model.add(LSTM(32, stateful=True))
        model.add(Dense(data_dim, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        # Generate dummy training data
        
        # x_train = np.random.random((1000, timesteps, data_dim))

        for cnt, (current_x_gen , next_x_gen) in enumerate(self.data_generator(data_dim, timesteps)):
            #print("started loop")
            x_train = current_x_gen

            # y_train = np.random.random((1000, num_classes))
            try:
                y_train = next_x_gen[:,0]
            except IndexError:
                print(y_train.shape)
                continue

            #print("training")

            # Generate dummy validation data
            # x_val = np.random.random((batch_size, timesteps, data_dim))
            # y_val = np.random.random((batch_size, num_classes))

            model.fit(x_train, y_train,
                batch_size=batch_size, epochs=3, shuffle=False)

            #print("finished batch")
            if cnt % 100 == 0:

                model.save("poetry_lstm_" + str(cnt) + ".h5")

    def create_poem(self):
        model = load_model('poetry lstm0')
        self.load_token_dict()

        rand_char = self.token_id_dict["l"]
        input_dat = np.array(rand_char)
        data_point = np_utils.to_categorical(input_dat, num_classes=len(self.token_id_dict) + 1)
        data_point = data_point[np.newaxis , :]

        tiled_data_point = np.tile(data_point,(100, 100 , 1))
        #print(tiled_data_point.shape)

        pred = model.predict(tiled_data_point, batch_size=100)
    
alexa_poet = Poet()
#alexa_poet.create_token_dict()

import time
t0 = time.time()

#alexa_poet.train_lstm()
alexa_poet.train_lstm()

t1 = time.time()
print("elapsed = " + str(t1 - t0) + "s")
                          

    

    
    

    
