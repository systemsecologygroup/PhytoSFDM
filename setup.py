#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' PhytoSFDM Package setup file'''

import re
from setuptools import setup
from os import path

version = re.search('^__version__\s*=\s*"(.*)"', open('phytosfdm/Example/example.py').read(),re.M).group(1)
 
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst')) as f:
    long_description = f.read().decode("utf-8")
 
setup(
    name="PhytoSFDM",
    version=version,
    author="Esteban Acevedo-Trejos",
    author_email="esteban.acevedo@zmt-bremen.de",
    license="GPLv2",
    description="PhytoSFDM is a modelling framework to quantify phytoplankton community structure and functional diversity",
    long_description=long_description,
    url="",
    packages=["phytosfdm", "phytosfdm.Example", "phytosfdm.SizeModels", "phytosfdm.EnvForcing"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Education",
        "Topic :: Scientific/Engineering"
    ],
    entry_points={"console_scripts": ['PhytoSFDM_example = phytosfdm.Example.example:main']},
    include_package_data=True,
    install_requires=['numpy>=1.9.2','scipy>=0.15.1','sympy>=0.7.6.1','matplotlib>=1.4.3'],
    
)