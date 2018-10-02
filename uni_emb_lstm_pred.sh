#!/bin/bash

LC_ALL=en_US.UTF-8 PYTHONPATH=`pwd` python src/uni_embed/emit_corpus_text.py --lstm ./exp/uni_embed/lstm/corpus.pred "$@"
