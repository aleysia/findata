from setuptools import find_packages, setup
import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='findata',
    packages=find_packages(),
    install_requires=install_requires,
    version='1.0.0',
    description='Package for downloading, cleaning, and modeling stock market data.',
    author='Virginia Chen'
)
