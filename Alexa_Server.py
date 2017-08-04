from flask import Flask
from flask import Blueprint, render_template
from flask_ask import Ask, statement, question

import os


from flask import Flask
from flask_ask import Ask, statement, question
import requests
import time
import unidecode
import json


class AppServer():
    def __init__(self):
        self.alexa_modules = list()
        self.module_id_file_name = "alexa.skill"

        self.ask_objects = list()

        for entry in os.scandir('.'):
            if entry.is_dir():
                for test_file in os.scandir(entry.path):
                    if test_file.name == self.module_id_file_name:
                        self.alexa_modules.append(entry)

        self.app = Flask(__name__)
        for module in self.alexa_modules:


            listen_url = Blueprint(("/" + str(module.name)), __name__)
            @listen_url.route("/" + module.name)
            def show():
                try:
                    return str(module.name)
                except TemplateNotFound:
                    abort(404)




            self.app.register_blueprint(listen_url)
            self.ask_objects.append(Ask(blueprint =  listen_url))


testTree = AppServer()
print(list(testTree.app.url_map.iter_rules()))
if __name__ == '__main__':
    testTree.app.run(debug=True)








    