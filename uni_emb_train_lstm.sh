#!/bin/bash

LC_ALL=en_US.UTF-8 PYTHONPATH=`pwd` python src/uni_embed/train_model.py --lstm ./exp/uni_embed/lstm/model.h5 "$@"
