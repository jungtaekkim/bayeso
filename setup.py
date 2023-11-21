from setuptools import setup
import os
import sys
from pathlib import Path

if (sys.version_info.major == 3 and sys.version_info.minor == 7) or\
    (sys.version_info.major == 3 and sys.version_info.minor == 8) or\
    (sys.version_info.major == 3 and sys.version_info.minor == 9) or\
    (sys.version_info.major == 3 and sys.version_info.minor == 10) or\
    (sys.version_info.major == 3 and sys.version_info.minor == 11):
    print(f'[SETUP] bayeso supports Python {sys.version_info.major}.{sys.version_info.minor} version in this system.')
else:
    sys.exit(f'[ERROR] bayeso does not support Python {sys.version_info.major}.{sys.version_info.minor} version in this system.')

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

path_requirements = 'requirements.txt'
list_packages = [
    'bayeso',
    'bayeso.bo',
    'bayeso.gp',
    'bayeso.tp',
    'bayeso.trees',
    'bayeso.wrappers',
    'bayeso.utils'
]

with open(path_requirements) as f:
    required = f.read().splitlines()

setup(
    name='bayeso',
    version='0.5.5',
    author='Jungtaek Kim',
    author_email='jtkim@postech.ac.kr',
    url='https://bayeso.org',
    license='MIT',
    description='Simple, but essential Bayesian optimization package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=list_packages,
    python_requires='>=3.7, <4',
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
