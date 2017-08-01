import nltk
from nltk import word_tokenize

import numpy as np

from collections import Counter
from collections import defaultdict

class Poet:

    def __init__(self, path):

        """
        Decode text file from bytes to string, then make lowercase for processing.
        Then, initialize variables.

        Parameters
        ----------
        path : str
            File path to text, ex: poems.txt
        
        """

        path_to_poems = path

         with open(path_to_poems, "rb") as f:
            self.poems = f.read().decode().lower()

        self.counter = Counter(self.poems)
        
        pass

    def load_text(self, path):

        """
        Decode text file from bytes to string, then make lowercase for processing.

        Parameters
        ----------
        path : str
            File path to text, ex: poems.txt
        
        """

        path_to_poems = path

        with open(path_to_poems, "rb") as f:
            self.poems = f.read().decode().lower()

        return None
    
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

        total = sum(cnt for _, cnt in counts)
        frequencies = [ (char, cnt/total) for char, cnt in counts ]

        # sanity check: confirm that the frequencies total to 1

        print(sum(freq for _, freq in frequencies))

        return frequencies

    def tokenize(self):

        tokens = self.poems.split() # splits text by spaces

        return tokens

    def train_lm(text, n):
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

        # initialize the model
        model = defaultdict(Counter)

        # create the initial padded history
        history = "~"*(n-1)

        for i in range(len(text)):
            next_char = text[i] # get the next character in the text
            model[history][next_char] += 1 # given the history, update the counter

            history = str( history[ - (n-2):] + next_char ) # revise the history

        # use the normalize function to convert the history -> char-count dict to
        # a history -> char-frequency dict

        for x in model:
            model[x] = normalize(model[x])

        return model

    def generate_letter(lm, history):
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
        
        if not history in lm:
            return "~"

        letters, probabilities = unzip(lm[history])

        i = np.random.choice(letters, p=probabilities)

        return i

    def generate_text(lm, n, nletters=100):
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
        history = "~" * (n-1)

        text = []

        for i in range(nletters):
            
            c = generate_letter(lm, history)
            text.append(c)

            history = history[1:]

        return "".join(text) # list to str
            
                          

    

    
    

    
