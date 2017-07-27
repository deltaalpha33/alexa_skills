import feedparser
import justext
import pickle
import requests
import sys

import nltk
from nltk.tokenize import word_tokenize

class Entity_Database:

    """
    Class stores entities created by searching through documents.
    Uses counter to sort most common entities in document, for searching.

    { entity : doc id }


         ent1     ent2    ent3    ...    entn

    ent1   None    1       2              X

    ent2   1       None    0              Y

    ent3   2       0       None           Z

    ...

    entn   X       Y       Z              None

    Entities co-ocurring w/ Counter keeping track
    """

    def __init__(self, links):

        """
        Creates global variables database, document list
        
        Parameters
        ----------
        links : list, str
                Array of links from which to get document text
        """
        
        self.database = {}
        self.documents = {} # dict[id] = [document]

        id = 1
        for l in links:
            documents[id] = get_text(l)
            id += 1
        pass

    def get_text(link):
        """
        Convert HTML to plain text, while removing boilerplate.
        
        Parameters
        ----------
        link : HTML page
               Converted to text separated by paragraph

        Returns
        -------
        text : text
               Converted from HTML page, separated by paragraph
        """
        
        response = requests.get(link)
        paragraphs = justext.justext(response.content, justext.get_stoplist("English"))
        text = "\n\n".join([p.text for p in paragraphs if not p.is_boilerplate])

        return text

    def collect(url, filename):
        # read RSS feed
        d = feedparser.parse(url)

        # grab each article
        texts = {}

        for entry in d["entries"]:
            
            link = entry["link"]
            print("downloading: " + link)

            text = get_text(link)
            texts[link] = text

        # saves texts into a pickle
        pickle.dump(texts, open(filename, "wb"))
        pass

    def database(self):
        """
        Returns
        -------
        database : dict[entity, ids]
        """

        return database

    def extract_entities(self):

        # first, tokenize the text
        tokens = nltk.word_tokenize(raw_text) # returns list of tokenized words

        # label the parts of speech
        parts_speech = nltk.pos_tag(tokens)

        # identify the named entities from the parts of speech
        entities = nltk.ne_chunk(parts_speech, binary=True)

        return entities

    def find_in_doc(self, doc, ent):
        """
        Did the entity appear in the given document?

        Returns
        -------
        count : Counter
                How many times it occurred in the document

        """
        pass

    def rank_documents(self, ent):
        """
        Rank documents by how often the entity occured in each - in descending order of most to least often.
        """

    def add_document(self, docs):
        pass

    def remove(self):
        pass

    def search(self):
        pass

    
