from setuptools import setup
import os
import sys

if sys.version_info < (3, 0):
    sys.exit('Python < 3.0 is not supported.')

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='bayeso',
    version='0.2.3',
    author='Jungtaek Kim',
    author_email='jtkim@postech.ac.kr',
    url='https://github.com/jungtaekkim/bayeso',
    license='MIT',
    description='Bayesian optimization package',
    packages=['bayeso', 'bayeso.utils'],
    python_requires='>=3',
    install_requires=required,
)
