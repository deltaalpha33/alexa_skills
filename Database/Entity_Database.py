import feedparser
import justext
import pickle
import requests
import sys

import nltk
from nltk.tokenize import word_tokenize

from Search_Engine import Search_Engine

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

    def __init__(self, rss_links = list()):

        """
        Creates instance variables database, document list
        
        Parameters
        ----------
        rss_links : list, str
                Array of rss urls to take news from
        """
        
        self.entities = dict()
        self.documents = dict() # dict[id] = [document]

        self.search_engine = Search_Engine()

        self.rss_links = rss_links
        self.scrape_from_rss()
        self.extract_entities()

    def scrape_from_rss(self):
        """
        Scrapes the Rss feeds in self.rss_links for news information and updates self.databse
        """
        for link in self.rss_links:
            self.collect(link)
        pass

    def collect(self, url):
        """
        Collects documents from url, updates self.documents
     
        Parameters
        ----------
        url : str
              URL from which to take news articles
        """
        
        # read RSS feed
        d = feedparser.parse(url) #get dictionary of links and entries


        for entry in d["entries"]:
            
            link = entry["link"] #get link for an entry
            #print("downloading: " + link)

            text = get_text(link) #downloads from link and parses text
            self.documents[link] = text

        # saves texts into a pickle
        #pickle.dump(texts, open(filename, "wb"))
        pass

    def get_text(self, link):
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

    def extract_entities(self):
        """
        Searches through documents in self.documents, and stores all entities found into self.entities
        
        """
        
        for link, raw_text in self.documents.items():
            # first, tokenize the text
            tokens = nltk.word_tokenize(raw_text) # returns list of tokenized words

            # label the parts of speech
            parts_speech = nltk.pos_tag(tokens)

            # identify the named entities from the parts of speech
            document_entities = nltk.ne_chunk(parts_speech, binary=True)

            self.entities[link] = document_entities
        pass
    
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

    
