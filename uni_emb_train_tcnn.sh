#!/bin/bash

LC_ALL=en_US.UTF-8 PYTHONPATH=`pwd` python src/uni_embed/train_model.py --tcnn ./exp/uni_embed/tcnn/model.h5 "$@"
