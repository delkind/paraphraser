import sys

import torch
from InferSent.models import InferSent
import codecs
import csv
import re

from keras.utils.data_utils import get_file
import h5py

URL_ROOT = "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_"
CSV_EXT = ".csv"


class BibleCorpora:
    def __init__(self, files, base_url=URL_ROOT, suffix=CSV_EXT, test_split=0.1, validation_split=0.1):
        corpora = {}
        for file in files:
            corpus = {}
            with open(get_file(file, base_url + file + suffix, cache_dir='/tmp/bible.cache/'), "rb") as webfile:
                for idx, row in enumerate(csv.reader(codecs.iterdecode(webfile, 'utf-8'))):
                    if idx > 0:
                        corpus[tuple(int(v) for v in row[:-1])] = self.normalize(row[4])
            corpora['<{}>'.format(file)] = corpus

        keysets = [set(ks.keys()) for ks in corpora.values()]
        intersection = [s for s in sorted(list(keysets[0].intersection(*keysets[1:])))
                        if len([corp[s] for corp in corpora.values() if len(corp[s]) == 0]) == 0]

        self.corpora = {file: [corpus[key] for key in intersection] for (file, corpus) in corpora.items()}

    def normalize(self, sentence):
        sentence = re.sub('[^ a-zA-Z0-9.,:;!\?\'{}]', ' ', sentence.lower())
        sentence = sentence.strip()
        psalms = re.findall('psalm [0-9]+', sentence)
        if len(psalms) > 0:
            sentence = sentence.split(psalms[0])[0]
        sentence = re.sub(r'\{(.*?)\}', '', sentence)
        sentence = re.sub('[0-9]+ <COLON> [0-9]+', '', sentence)

        return sentence


def create_bible_embeddings(path):
    print("Creating InferSent embeddings for bible corpora...")
    infersent = InferSent({'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                           'pool_type': 'max', 'dpout_model': 0.0, 'version': 2})
    infersent.load_state_dict(torch.load('InferSent/encoder/infersent2.pkl'))
    infersent.set_w2v_path('InferSent/dataset/fastText/crawl-300d-2M.vec')

    print("Loading corpora...")
    dataset = BibleCorpora(["bbe", "ylt"], URL_ROOT, CSV_EXT)
    infersent.build_vocab(dataset.corpora['<bbe>'] + dataset.corpora['<ylt>'], tokenize=True)
    print("Creating embeddings for BBE corpus...")
    encoded_bbe = infersent.encode(dataset.corpora['<bbe>'], tokenize=True)
    print("Creating embeddings for YLT corpus...")
    encoded_ylt = infersent.encode(dataset.corpora['<ylt>'], tokenize=True)
    print("Saving embeddings to {}...".format(path))
    with h5py.File(path, "w") as ds:
        ds.create_dataset("bbe", data=encoded_bbe)
        ds.create_dataset("ylt", data=encoded_ylt)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} <embeddings_dest_path>")
    else:
        create_bible_embeddings(sys.argv[1])
    pass
