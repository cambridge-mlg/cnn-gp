#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from setuptools import setup

setup(name='cnn_gp',
      version="0.1",
      author="Adrià Garriga-Alonso, Laurence Aitchison",
      author_email="adria.garriga@gmail.com",
      description="CNN-GPs in Pytorch",
      license="BSD License 2.0",
      url="http://github.com/cambridge-mlg/cnn-gp-pytorch",
      ext_modules=[],
      packages=["cnn_gp"],
      install_requires="""
          numpy>=1.10.0
          torch>=1.1.0
          torchvision>=0.2.0
          tqdm>=4.32
      """.split())
