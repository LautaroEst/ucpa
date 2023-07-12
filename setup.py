import os
from setuptools import setup

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        fcontent = f.read()
    return fcontent
    

setup(
    name = "ucpa",
    version = "0.0.1",
    author = "Lautaro Estienne",
    author_email = "lestienne@fi.uba.ar",
    description = ("Code to work with the UCPA method"),
    keywords = "UCPA SUCPA calibration",
    url = "https://github.com/LautaroEst/ucpa",
    packages=['ucpa'],
    long_description=read('Readme.md')
)
