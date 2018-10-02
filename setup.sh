#!/bin/bash
git submodule update --init --recursive
virtualenv --python=python3 .env
source ./.env/bin/activate
pip install -r requirements.txt
cd keras-tcn
pip install . --upgrade
LC_ALL=en_US.UTF-8 python setup.py

