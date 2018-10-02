#!/bin/bash

mkdir -p ./exp/uni_embed/tcnn
mkdir -p ./exp/uni_embed/lstm

LC_ALL=en_US.UTF-8 PYTHONPATH=`pwd` python src/uni_embed/download_models.py ./exp/uni_embed/lstm/model.h5 ./exp/uni_embed/tcnn/model.h5 ./exp/uni_embed/embeddings.h5
