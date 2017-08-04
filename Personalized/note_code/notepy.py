import numpy as np
import os
import shutil

class Note:
    def __init__(self,path,audio):
        self.path = path
        self.a = audio.Audio()
        self.db = []
    def add_to_db(self,note,to):
        self.db.append([note,to])
    def saveDBnp(self):
        """
        Saves a db to directory path
        """
        if len(self.db) == 0:
            return 0
        it = 0
        dirt = self.path
        prevname = self.db[0][1]
        for entr in self.db:
            ray, name = entr[0],entr[1]
            if name != prevname:
                prevname = name
                it = 0
            direc = os.path.join(dirt , name)

            if not os.path.exists(direc):
                os.makedirs(direc)


            direc = os.path.join(direc , "note" + str(it))

            np.savez(direc,ray=ray)
            it = it + 1
    def loadDBnp(self):
        """
        Loads a db from directory path.
        the folder at path path must be formated like such:
        Folders with names of the desired labels (ie: 'Daschel Cooper')
        Within them .npz files storing arrays named 'ray'
            (this naming and format is done automatically by saveDBnp)
        """
        dirt = self.path
        lstOfDirs = [x[0] for x in os.walk(dirt)][1:]
        splt = os.sep
        db = []

        for rootDir in lstOfDirs:
            print(rootDir)
            fileSet = set()



            for dir_, _, files in os.walk(rootDir):
                for fileName in files:
                    relDir = os.path.relpath(dir_, rootDir)
                    relFile = os.path.join(rootDir, fileName)
                    if not fileName.startswith('.'):
                        fileSet.add(relFile)

            for file in fileSet:
                vector = np.load(file)['ray']
                name = rootDir.split(splt)[1]
                db.append( (vector , name) )

        self.db = db
    def add_note(self, time, name):
        """
        record + store a note of length time for name
        """
        print("d5aefcace8face5e85ae8f5819234a")
        note = self.a.read_mic(time)
        self.db.append( [ note,name ] )
        self.saveDBnp()
    def read_notes(self, name):
        killMe = []
        for i,val in enumerate(self.db):
            note, nameDB = val[0],val[1]
            if nameDB == name:
                print("in")
                self.a.play_audio(note)
                killMe.append(i)
        self.db = [i for j, i in enumerate(self.db) if j not in killMe]
        #os.rmdir(os.path.join(self.path,name))
        shutil.rmtree(os.path.join(self.path,name), ignore_errors=True)

