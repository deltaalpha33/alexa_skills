from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import string
import time
from operator import itemgetter

class MySearchEngine():
    def __init__(self):
        
        self.raw_text = {} # Dict[str, str]: maps document id to original/raw text
        
        self.term_vectors = {} # Dict[str, Counter]: maps document id to term vector (counts of terms in document)
        
        self.doc_freq = Counter()  # Counter: maps term to count of how many documents contain term
        
        self.inverted_index = defaultdict(set) # Dict[str, set]: maps term to set of ids of documents that contain term
    
    # ------------------------------------------------------------------------
    #  INDEXING
    # ------------------------------------------------------------------------
    
    def strindex(self, text, val):
        """
        """
        if (text + val).index(val) == len(text):
            return -1

        return text.index(val)

    def tokenize(self, text):
        """ Converts text into tokens (also called "terms" or "words").
        
            This function should also handle normalization, e.g., lowercasing and 
            removing punctuation.
        
            For example, "The cat in the hat." --> ["the", "cat", "in", "the", "hat"]
        
            Parameters
            ----------
            text: str
                The string to separate into tokens.
        
            Returns
            -------
            list(str)
                A list where each element is a token.
        
        """
        # tokenize, change to lowercase, and filter out punctuation (using string.punctuation)
        
        return [y.lower() for y in nltk.word_tokenize(text) if self.strindex(string.punctuation,y) == -1] 


    def add(self, id, text):
        """ Adds document to index.
        
            Parameters
            ----------
            id: str
                A unique identifier for the document to add, e.g., the URL of a webpage.
            text: str
                The text of the document to be indexed.
        """
        # check if document is already in index, and raise error if so
        if id in self.raw_text:
            raise LookupError("Document already in index - no need to add it!")

        self.raw_text[id] = text
        
        tokens = self.tokenize(text)
        
        c = Counter(tokens)
        
        self.term_vectors[id] = c
        
        for vv in c:
            self.inverted_index[vv].update([id])
        
        storage = set()
        storage.update( c.elements() )
        for varK in storage:
            self.doc_freq[varK] += 1
        
    def remove(self, id):
        """ Removes document from index.
        
            Parameters
            ----------
            id: str
                The identifier of the document to remove from the index.
        """

        ids = self.raw_text.keys()
        
        # check if document exists and throw exception if not
        if id not in ids:
            raise LookupError("This document isn't in the index - no need to remove it!")

        # remove raw text for this document
        del self.raw_text[id]        

        c = self.term_vectors[id]
        
        for vv in self.term_vectors[id]:
            self.inverted_index[vv] -= set([id])

        # update document frequencies for terms found in this doc
        # i.e., counts should decrease by 1 for each (unique) term in term vector
        storage = set()
        storage.update( c.elements() )
        for varK in storage:
            self.doc_freq[varK] -= 1

        # remove term vector for this doc
        del self.term_vectors[id]

    def get(self, id):
        """ Returns the original (raw) text of a document.
        
            Parameters
            ----------
            id: str
                The identifier of the document to return.
        """
       
       ids = self.raw_text.keys()
       
       # check if document exists and throw exception if not
       if id not in ids:
            raise LookupError("This document isn't in the index!")

        return self.raw_text[id]  # return raw text
    
    def num_docs(self):
        """ Returns the current number of documents in index. 
        """
        
        return len(self.raw_text)

    # ------------------------------------------------------------------------
    #  MATCHING
    # ------------------------------------------------------------------------

    def get_matches_term(self, term):
        """ Returns ids of documents that contain term.
        
            Parameters
            ----------
            term: str
                A single token, e.g., "cat" to match on.
            
            Returns
            -------
            set(str)
                A set of ids of documents that contain term.
        """
        # must be lowercased so it can match the output of tokenizer
        
        term = term.lower()
        
        return self.inverted_index[term] # look up term in inverted index

    def get_matches_OR(self, terms):
        """ Returns set of documents that contain at least one of the specified terms.
        
            Parameters
            ----------
            terms: iterable(str)
                An iterable of terms to match on, e.g., ["cat", "hat"].
            
            Returns
            -------
            set(str)
                A set of ids of documents that contain at least one of the term.
        """
        
        s = set() # initialize set of ids to empty set

        # union ids with sets of ids matching any of the terms
        for term in terms:
            t = term.lower()
            s.update(self.inverted_index[t])
        return s
    
    def get_matches_AND(self, terms):
        """ Returns set of documents that contain all of the specified terms.
        
            Parameters
            ----------
            terms: iterable(str)
                An iterable of terms to match on, e.g., ["cat", "hat"].
            
            Returns
            -------
            set(str)
                A set of ids of documents that contain each term.
        """
        
        s = set()
        s.update(self.inverted_index[terms[0]]) # initialize set of ids to those that match first term

        # intersect with sets of ids matching rest of terms
        for term in terms:
            t = term.lower()
            s &= (self.inverted_index[t])
        return s
    
    def get_matches_NOT(self, terms):
        """ Returns set of documents that don't contain any of the specified terms.
        
            Parameters
            ----------
            terms: iterable(str)
                An iterable of terms to avoid, e.g., ["cat", "hat"].
            
            Returns
            -------
            set(str)
                A set of ids of documents that don't contain any of the terms.
        """
        
        ids = set(self.raw_text.keys())
        
        for term in terms:
            
            ids -= set(self.inverted_index[term.lower()])
        return ids

    # ------------------------------------------------------------------------
    #  SCORING
    # ------------------------------------------------------------------------
        
    def idf(self, term):
        """ Returns current inverse document frequency weight for a specified term.
        
            Parameters
            ----------
            term: str
                A term.
            
            Returns
            -------
            float
                The value idf(t, D) as defined above.
        """
        
        N = len(self.raw_text.keys())
        n = self.doc_freq[term]
        
        return np.log10(N/(1+n))
    
    def dot_product(self, tv1, tv2):
        """ Returns dot product between two term vectors (including idf weighting).
        
            Parameters
            ----------
            tv1: Counter
                A Counter that contains term frequencies for terms in document 1.
            tv2: Counter
                A Counter that contains term frequencies for terms in document 2.
            
            Returns
            -------
            float
                The dot product of documents 1 and 2 as defined above.
        """
        
        # iterate over terms of one document
        # if term is also in other document, 
        # then add their product (tfidf(t,d1) * tfidf(t,d2)) to a running total

        total = 0
        
        for t in tv1:
            total += self.idf(t) * tv1[t]*tv2[t] * self.idf(t)
        
        return total
    
    def length(self, tv):
        """ Returns the length of a document (including idf weighting).
        
            Parameters
            ----------
            tv: Counter
                A Counter that contains term frequencies for terms in the document.
            
            Returns
            -------
            float
                The length of the document as defined above.
        """
        
        total = 0
        for t in tv:
            total += (tv[t]*self.idf(t))**2
            
        return total ** 0.5
    
    def cosine_similarity(self, tv1, tv2):
        """ Returns the cosine similarity (including idf weighting).

            Parameters
            ----------
            tv1: Counter
                A Counter that contains term frequencies for terms in document 1.
            tv2: Counter
                A Counter that contains term frequencies for terms in document 2.
            
            Returns
            -------
            float
                The cosine similarity of documents 1 and 2 as defined above.
        """
        
        dot = self.dot_product(tv1,tv2)
        mags = self.length(tv1) * self.length(tv2)
        
        return dot/mags

    # ------------------------------------------------------------------------
    #  QUERYING
    # ------------------------------------------------------------------------

    def query(self, q, k=10):
        """ Returns up to top k documents matching at least one term in query q, sorted by relevance.
        
            Parameters
            ----------
            q: str
                A string containing words to match on, e.g., "cat hat".
        
            Returns
            -------
            List(tuple(str, float))
                A list of (document, score) pairs sorted in descending order.
                
        """
        
        # tokenize query
        tokens = self.tokenize(q)

        # get matches (just support OR style queries for now...)
        matches = self.get_matches_OR(tokens)
                
        # convert query to a term vector (Counter over tokens)
        c = Counter(tokens)
        
        scores = []
        # score each match by computing cosine similarity between query and document
        for m in matches:
            scores.append( ( m, self.cosine_similarity(c,self.term_vectors[m]) ) )
       
        scores.sort(key=itemgetter(1))
        scores = scores[::-1]
        
        # sort results and return top k
        return scores[:k]
        
    def advanced_query(self, q, k=10,require=None,exclude=None):
        """ Returns up to top k documents matching at least one term in query q, sorted by relevance.
        
            Parameters
            ----------
            q: str
                A string containing words to match on, e.g., "cat hat".
                +word means require word
                -word means require not having word
                
            
        
            Returns
            -------
            List(tuple(str, float))
                A list of (document, score) pairs sorted in descending order.
                
        """
        
        doAnd = False
        doNot = False
        
        if require != None:
            ands = self.get_matches_AND( self.tokenize(require) )
            doAnd = True
            
        if exclude != None:
            nots = self.get_matches_NOT( self.tokenize(exclude) )
            doNot = True
            
        # tokenize query
        tokens = self.tokenize(q)

        # get matches (just support OR style queries for now...)
        matches = self.get_matches_OR(tokens)
                
        if doAnd:
            fix_matches = (matches & ands) 
        if doNot:
            fix_matches = (matches & nots)
            
        # convert query to a term vector (Counter over tokens)
        c = Counter(tokens)
        
        scores = []
        # score each match by computing cosine similarity between query and document
        for m in fix_matches:
            scores.append( ( m, self.cosine_similarity(c,self.term_vectors[m]) ) )
        
        scores.sort(key=itemgetter(1))
        scores = scores[::-1]
        
        # sort results and return top k
        return scores[:k]
        
