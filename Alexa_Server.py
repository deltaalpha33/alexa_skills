from flask import Flask
from flask_ask import Ask, statement, question

import os

class AppTree():
    def __init__(self):
        self.alexa_modules = list()
        self.module_id_file_name = "alexa.skill"


        for entry in os.scandir('.'):
            if entry.is_dir():
                for test_file in os.scandir(entry.path):
                    if test_file.name == self.module_id_file_name:
                        self.alexa_modules.append(entry)
                        


testTree = AppTree()
print(testTree.alexa_modules)