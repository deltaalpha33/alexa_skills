from setuptools import setup, find_packages

def do_setup():
    setup(name="SpoontopSkills",
          version="0.0",
          author='Spoontop, Group 7',
          description='Final Cogworks project',
          platforms=['Windows', 'Linux', 'Mac OS-X'],
          packages=find_packages(),
          install_requires=['feedparser', 'justext', 'nltk', 'ngrok', 'sklearn', 'gensim', 'numba'])

if __name__ = "__main__":
    do_setup()
    


    
    
