#import nltk
#from nltk import word_tokenize

import numpy as np

from collections import Counter
from collections import defaultdict

import os
import io
import glob

import re, string
import pickle

#from keras.models import Sequential
#from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop



from sklearn.decomposition import TruncatedSVD, randomized_svd
from gensim.models.keyedvectors import KeyedVectors
from collections import defaultdict, Counter
import codecs
from nltk.tokenize import word_tokenize
import time

from numba import njit


def generate_sorted_words(tokens):
    """ Create list of unique words sorted by count in descending order
        
        Parameters
        ----------
        tokens: list(str)
            A list of tokens (words), e.g., ["the", "cat", "in", "the", "in", "the"]
        
        Returns
        -------
        list(str)
            A list of unique tokens sorted in descending order, e.g., ["the", "in", cat"]
        
    """
    # SOLUTION
    counter = Counter(tokens)
    words = [word for word, count in counter.most_common()]
    return words

def generate_word2code(sorted_words):
    """ Create dict that maps a word to its position in the sorted list of words
    
        Parameters
        ---------
        sorted_words: list(str)
            A list of unique words, e.g., ["b", "c", "a"]
        
        Returns
        -------
        dict[str, int]
            A dictionary that maps a word to an integer code, e.g., {"b": 0, "c": 1, "a": 2}
        
    """
    # SOLUTION
    word2code = {w : i for i, w in enumerate(sorted_words)}
    return word2code

def convert_tokens_to_codes(tokens, word2code):
    """ Convert tokens to codes.
    
        Parameters
        ---------
        tokens: list(str)
            A list of words, e.g., ["b", "c", "a"]
        word2code: dict[str, int]
            A dictionary mapping words to integer codes, e.g., {"b": 0, "c": 1, "a": 2}
        
        Returns
        -------
        list(int)
            A list of codes corresponding to the input words, e.g., [0, 1, 2].
    """
    # SOLUTION
    return [word2code[token] for token in tokens]


def reduce(X, n_components, power=0.0):
    U, Sigma, VT = randomized_svd(X, n_components=n_components)
    # note: TruncatedSVD always multiplies U by Sigma, but can tune results by just using U or raising Sigma to a power
    return U * (Sigma**power)

@njit
def generate_word_by_context(codes, max_vocab_words=1000, max_context_words=1000, context_size=2, weight_by_distance=False):
    """ Create matrix of vocab word by context word (possibly weighted) co-occurrence counts.
    
        Parameters
        ----------
        codes: list(int)
            A sequence of word codes.
        max_vocab_words: int
            The max number of words to include in vocabulary (will correspond to rows in matrix).
            This is equivalent to the max word code that will be considered/processed as the center word in a window.
        max_context_words: int
            The max number of words to consider as possible context words (will correspond to columns in matrix).
            This is equivalent to the max word code that will be considered/processed when scanning over contexts.
        context_size: int
            The number of words to consider on both sides (i.e., to the left and to the right) of the center word in a window.
        weight_by_distance: bool
            Whether or not the contribution of seeing a context word near a center word should be 
            (down-)weighted by their distance:
            
                False --> contribution is 1.0
                True  --> contribution is 1.0 / (distance between center word position and context word position)
            
            For example, suppose ["i", "am", "scared", "of", "dogs"] has codes [45, 10, 222, 25, 88]. 
            
            With weighting False, 
                X[222, 25], X[222, 10], X[222, 25], and X[222, 88] all get incremented by 1.
            
            With weighting True, 
                X[222, 25] += 1.0/2 
                X[222, 10] += 1.0/1 
                X[222, 25] += 1.0/1
                X[222, 88] += 1.0/2
        
        Returns
        -------
        (max_vocab_words x max_context_words) ndarray
            A matrix where rows are vocab words, columns are context words, and values are
            (possibly weighted) co-occurrence counts.
    """
    
    """
    pseudo-code:
    
    slide window along sequence
    if code of center word is < max_vocab_words
        for each word in context (on left and right sides)
            if code of context word < max_context_words
                add 1.0 to matrix element in row of center word and column of context word
                    or
                add 1.0 / (distance from center to context)
    
    example: assume context_size is 2 (i.e., 2 words to left and 2 words to right)
    
      "a" "a" "b" "c" "c" "c" "c" "a" "b" "c"   # sequence of words
       1   1   2   0   0   0   0   1   2   0    # sequence of word codes
       0   1   2   3   4   5   6   7   8   9    # position in sequence
      [        ^        ]                       # first window: centered on position 2; center word has code 2
          [        ^        ]                   # second window: centered on position 3; center word has code 0
                       ...                 
                          [        ^        ]   # last window: centered on position 7; center word has code 1
    """
    
    # initialize matrix (with dtype="float32" to reduce required memory)
    
    # SOLUTION
    X = np.zeros((max_vocab_words, max_context_words))

    # slide window along sequence and count "center word code" / "context word code" co-occurrences
    # Hint: let main loop index indicate the center of the window
    
    # SOLUTION
    for i in range(context_size, len(codes) - context_size):
#         if i % 100000 == 0:
#             print("i = " + str(i) + ": " + str(1.0 * i / len(codes)) + "%")

        center_code = codes[i]
        if center_code < max_vocab_words:
            # left side
            for j in range(1, context_size + 1):
                context_code = codes[i - j]
                if context_code < max_context_words:
                    value = 1.0
                    if weight_by_distance:
                        value = 1.0 / j
                    X[center_code, context_code] += value
            # right side
            for j in range(1, context_size + 1):
                context_code = codes[i + j]
                if context_code < max_context_words:
                    value = 1.0
                    if weight_by_distance:
                        value = 1.0 / j
                    X[center_code, context_code] += value

    return X




class Poet:

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

        # initialize the model
        self.model = defaultdict(Counter)
        self.lm = defaultdict(Counter)

        self.n = 5

        self.word_descriptors = None

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
        alpha_split = re.compile("([A-Z][^A-Z]*)*")
        number_regex = re.compile('[[0-9]+]|[IVXLCM]+')
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

                

    def create_bag_of_words(self):
        """


        """
        punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

        file_paths = glob.glob(self.sanitized_poetry_dataset_dir +'*.txt')
        word_counter = Counter()
        for file_path in file_paths:

            with open(file_path, 'r') as f:
                document = f.read()
                document = punc_regex.sub('', document)
                document_words = document.split()
                word_counter.update(document_words)


        with open(self.sanitized_poetry_dataset_dir + "bag_of_words.pkl", 'wb') as f:
            pickle.dump(word_counter, f, pickle.HIGHEST_PROTOCOL)

    def load_bag_of_words(self):
        with open(self.sanitized_poetry_dataset_dir + "bag_of_words.pkl", 'rb') as f:
            self.bag_of_words = pickle.load(f)


    def learn_documents_ngram(self):
        """


        """
        punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
        space_regex = re.compile(' +')

        file_paths = glob.glob(self.sanitized_poetry_dataset_dir +'*.txt')
        document_counts = list()
        for file_path in file_paths:

            with open(file_path, 'r') as f:
                document = f.read()
                document = punc_regex.sub('', document)
                document = space_regex.sub(' ', document) #needs ' ' <----- space so it replaces instead of deletes

                char_counter = self.train_lm(document[1:])
                document_counts.append(char_counter)
        self.save_model_ngram()

    def save_model_ngram(self):
        with open(self.sanitized_poetry_dataset_dir + "n_gram_model.pkl", 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def load_model_ngram(self):
        with open(self.sanitized_poetry_dataset_dir + "n_gram_model.pkl", 'rb') as f:
            self.model = pickle.load(f)

    def learn_documents_word_gram(self):
        file_paths = glob.glob(self.sanitized_poetry_dataset_dir +'*.txt')
        document_counts = list()
        for file_path in file_paths:

            with open(file_path, 'r') as f:
                document = f.read()
                document = document.split(" ")

                word_counter = self.train_lm(document[1:])
                document_counts.append(word_counter)
        self.save_model_word_gram()

    def save_model_word_gram(self):
        with open(self.sanitized_poetry_dataset_dir + "word_gram_model.pkl", 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def load_model_word_gram(self):
        with open(self.sanitized_poetry_dataset_dir + "word_gram_model.pkl", 'rb') as f:
            self.model = pickle.load(f)
    
    def unzip(self, pairs):
        """
        Splits list of pairs (tuples) into separate lists.
        Ex: pairs = [("a", 1), ("b", 2)] -> ["a", "b], [1, 2]

        Parameters
        ----------
        pairs : list of tuples

        Returns
        -------
        tuple

        """

        return tuple(zip(*pairs))

    def char_count(self):

        letters = "abcdefghijklmnopqrstuvwxyz"

        counts = [ (char, cnt) for char, cnt in self. counter.most_common() if char in letters ]

        return counts

    def normalize(self, counts):
        """
        Converts counts to a list of letter-frequency pairs.

        Parameters
        ----------
        counts : Counter

        Returns
        -------
        list tuples

        """
        total = sum(cnt for cnt in counts.values())
        frequencies = [ (char, cnt/total) for char, cnt in counts.items()]

        # sanity check: confirm that the frequencies total to 1

        #print(sum(freq for _, freq in frequencies))

        return frequencies

    def nomalize_model(self):
        for cnt in self.model:
            self.lm[cnt] = self.normalize(self.model[cnt])

    def train_lm(self, text):
        """
        Train character-based n-gram language model.

        This will learn: given a sequence of n-1 characters, what the probability distribution is for the n-th character in the sequence.

        Parameters
        ----------
        text : str

        n : int
            Length of n-gram to analyze

        Returns
        -------
        A dictionary that maps histories to lists of pairs

        """



        # create the initial padded history
        history = "~"*(self.n-1)

        for i in range(len(text)):
            next_char = text[i] # get the next character in the text
            self.model[history][next_char] += 1 # given the history, update the counter

            history = str( history[ - (self.n-2):] + next_char ) # revise the history

        # use the normalize function to convert the history -> char-count dict to
        # a history -> char-frequency dict

        # for x in model:
        #     model[x] = normalize(model[x])

    def create_word_descriptors(self):
        self.load_bag_of_words()
        important_words = self.bag_of_words.most_common()[1000:90000]
        file_paths = glob.glob(self.sanitized_poetry_dataset_dir +'*.txt')
        tokens = list()
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                tokens.extend(f.read().split())

        #word embeddings
        tokens = [token for token in tokens if token in important_words]
        sorted_words = generate_sorted_words(tokens)
        word2code = generate_word2code(sorted_words)
        codes = convert_tokens_to_codes(tokens, word2code)
        # CAUTION: Think about how big of a matrix will be created...

        # how many words to keep in vocabulary (will have one row per vocab word)
        #max_vocab_words = len([word for word, count in alexa_poet.bag_of_words.most_common() if count > 10])
        max_vocab_words = len(important_words)

        # how many words to treat as potential context words (will have one column per context word)
        max_context_words = 5000


        t0 = time.time()
        X_wiki = generate_word_by_context(codes, 
                                          max_vocab_words=max_vocab_words, 
                                          max_context_words=max_context_words, 
                                          context_size=4,
                                          weight_by_distance=True)
        t1 = time.time()
        print("elapsed = " + str(t1 - t0) + "s")


        # apply log to raw counts (which has been shown to improve results)
        X_log = np.log10(1 + X_wiki, dtype="float32")

        t0 = time.time()
        d = 200
        my_vectors = reduce(X_log, n_components=d)
        t1 = time.time()
        print("elapsed " + str(t1 - t0) + "s")


        # save in word2vec format (first line has vocab_size and dimension; other lines have word followed by embedding)
        with codecs.open("word2vec.txt", "w", "utf-8") as f:
            f.write(str(max_vocab_words) + " " + str(d) + "\n")
            
            for i in range(max_vocab_words):
                f.write(sorted_words[i] + " " + " ".join([str(x) for x in my_vectors[i,:]]) + "\n")

        # with open(self.sanitized_poetry_dataset_dir + "word_descriptors.pkl", 'wb') as f:
        #     pickle.dump(my_vectors, f, pickle.HIGHEST_PROTOCOL)


        # # save in word2vec format (first line has vocab_size and dimension; other lines have word followed by embedding)
        # with codecs.open("test_vector.txt", "w", "utf-8") as f:
        #     f.write(str(len(sorted_words)) + " " + str(d) + "\n")
    def load_word_descriptors(self):

        # load back in
        with open(self.sanitized_poetry_dataset_dir + "word_descriptors.pkl", 'rb') as f:
            self.word_descriptors =  pickle.load(f)

    def generate_letter(self, history):
        """
        Randomly picks a letter according to the probability distribution associated with the given history.

        Parameters
        ----------
        lm : dict[ str, tuple[str, float] ]
            The n-gram language model

        history : str
            A string of length (n-1) to use as context for generating the next character.

        Returns
        -------
        str
            The predicted character given the history.
            "~" if there is no history.

        """
        
        if not history in self.lm:
            return "~"

        letters, probabilities = self.unzip(self.lm[history])

        i = np.random.choice(letters, p=probabilities)

        return i

    def generate_text(self, nletters=1000):
        """
        Randomly generates nletters of text with n-gram language model lm.

        Parameters
        ----------
        lm : dict[ str, tuple[str, float] ]
            The n-gram language model

        n : int
            The order of n-gram model

        nletters : int
            Number of letters to randomly generate

        Returns
        -------
        str
            Model-generated text

        """
        # pad the history
        history = "~" * (self.n-1)

        text = history

        for i in range(nletters):
            
            text += self.generate_letter(history)

            history = str(text[-self.n + 1:])

        return "".join(text) # list to str

    def get_longest_string(self):
        file_paths = glob.glob(self.sanitized_poetry_dataset_dir +'*.txt')
        longest_token_len = 0
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                doc = f.read().split()
                local_max_token_size = len(max(doc, key=len))
                if local_max_token_size > longest_token_len:
                    longest_token_len = local_max_token_size

                        
        print("longest string is " + str(len(max(doc, key=len))))
        return longest_token_len

    def train_lstm(self):
        phrase_size = 3
        


        



alexa_poet = Poet()
alexa_poet.load_bag_of_words()
alexa_poet.train_lstm()
#alexa_poet.load_word_descriptors()
#print(alexa_poet.word_descriptors["is"])
#alexa_poet.load_text("poetry_dataset/")
#alexa_poet.learn_documents_ngram()
#alexa_poet.load_model_ngram()
#alexa_poet.nomalize_model()
#print(alexa_poet.lm["~~"])
#print(alexa_poet.generate_letter("~~"))
#print(alexa_poet.generate_text())

#recurrent.recurrent([0,0,0], [1,2,3])


# as the first layer in a Sequential model
# model = Sequential()
# model.add(LSTM(32, input_shape=(10, 64)))

#alexa_poet.load_bag_of_words()
#print(len([word for word, count in alexa_poet.bag_of_words.most_common() if count > 10]))
            
                          

    

    
    

    
