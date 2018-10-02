#!/bin/bash

LC_ALL=en_US.UTF-8 PYTHONPATH=`pwd` python src/uni_embed/calc_bleu.py "$@"
