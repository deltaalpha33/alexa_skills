import nltk, pickle as plk
from nltk.tokenize import word_tokenize

class Entity_Database:

    """
    Class stores entities created by searching through documents.
    
    """

    def __init__(self, db):

        """
        Creates global variable database, a dictionary of [entity, ids]
        
        Parameters
        ----------
        db : dict[entity, ids]

        @TODO: Is the input a dictionary already made, or does the database
        create the dict from a list of ids and entities?

        """
        
        self.database = db
        pass

    def database(self):
        """
        Returns
        -------
        database : dict[entity, ids]
        """

        return database
    

    def add(self):
        pass

    def remove(self):
        pass

    def search(self):
        pass

    
