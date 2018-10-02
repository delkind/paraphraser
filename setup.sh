#!/bin/bash
virtualenv --python=python3 .env
pip install -r requirements.txt
cd keras-tcn
pip install . --upgrade
LC_ALL=en_US.UTF-8 python setup.py

