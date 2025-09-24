#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='lychee-smore',
      version='0.1.0',
      description='Official scripts for building SmoRe',
      author='KugaMaxx',
      author_email='KugaMaxx@outlook.com',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
)
