#!/bin/bash
curl -Lo InferSent/encoder/infersent1.pkl https://s3.amazonaws.com/senteval/infersent/infersent1.pkl
curl -Lo InferSent/encoder/infersent2.pkl https://s3.amazonaws.com/senteval/infersent/infersent2.pkl
mkdir InferSent/dataset/GloVe
curl -Lo InferSent/dataset/GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip InferSent/dataset/GloVe/glove.840B.300d.zip -d InferSent/dataset/GloVe/
mkdir InferSent/dataset/fastText
curl -Lo InferSent/dataset/fastText/crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
unzip InferSent/dataset/fastText/crawl-300d-2M.vec.zip -d InferSent/dataset/fastText/
cd InferSent/dataset
./get_data.bash
cd ../..
