#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='hacktrick_rl',
      version='0.0.1',
      description='This package has shared components.',
      packages=find_packages(),
      install_requires=[
        'GitPython',
        'memory_profiler',
        'sacred',
        'pymongo',
        'dill',
        'matplotlib',
        'requests',
        'pygame',
        'numpy',
        'seaborn==0.9.0',
        'ray[rllib]==0.8.5'
      ],
    )
