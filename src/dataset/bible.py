import codecs
import csv
import random
import re
from abc import abstractmethod, ABCMeta
import numpy as np

import keras
from keras.utils.data_utils import get_file
from collections import Counter

MIN_FREQ = 15
URL_ROOT = "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_"
CSV_EXT = ".csv"
OOV = '<OOV>'
MAX_SENTENCE_LENGTH = 60


def multi_replace(string, replacements, ignore_case=False):
    """
    Given a string and a dict, replaces occurrences of the dict keys found in the
    string, with their corresponding values. The replacements will occur in "one pass",
    i.e. there should be no clashes.
    :param str string: string to perform replacements on
    :param dict replacements: replacement dictionary {str_to_find: str_to_replace_with}
    :param bool ignore_case: whether to ignore case when looking for matches
    :rtype: str the replaced string
    """
    rep_sorted = sorted(replacements, key=lambda s: len(s[0]), reverse=True)
    rep_escaped = [re.escape(replacement) for replacement in rep_sorted]
    pattern = re.compile("|".join(rep_escaped), re.I if ignore_case else 0)
    return pattern.sub(lambda match: replacements[match.group(0)], string)


class Dataset(metaclass=ABCMeta):
    def __init__(self):
        self.mapping = {}
        self.frequencies = {}

    def create_mapping(self, corpora):
        segs = [seg for corpus in corpora.values() for sentence in corpus for seg in sentence]
        frequencies = Counter(segs).items()
        oov = [c for (_, c) in frequencies if c <= MIN_FREQ]
        self.frequencies = {k: l for (k, l) in frequencies if l > MIN_FREQ or k == OOV}
        self.frequencies[OOV] = sum(oov)
        self.mapping = {seg: num for (num, seg) in enumerate(self.frequencies.keys())}
        markers = ['start', 'end'] + list(corpora.keys())
        for m in markers:
            self.mapping['<{}>'.format(m)] = len(self.mapping)

    def tokenize(self, corpora):
        self.create_mapping(corpora)
        oov = self.mapping[OOV]
        return {key: [[self.mapping.get(seg, oov) for seg in sentence] for sentence in corpus]
                for (key, corpus) in corpora.items()}

    @staticmethod
    def normalize(sentence):
        sentence = re.sub('[^ a-zA-Z0-9.,:;!\?{}]', ' ', sentence.lower())
        rep = {'.': ' <DOT> ', ',': ' <COMMA> ', ';': ' <SEMICOLON> ', ':': ' <COLON> ', '!': ' <EXCL> ',
               '?': ' <QSTN> '}
        return multi_replace(sentence, rep)


class BibleDataset(Dataset):
    def __init__(self, base_url, files, suffix):
        super().__init__()
        self.tokenized_corpora = self.tokenize(BibleDataset.parse_csv(base_url, files, suffix))
        sentence_lengths = [len(s) for c in self.tokenized_corpora.values() for s in c]
        self.max_sentence_length = max(sentence_lengths)

    @staticmethod
    def parse_csv(base_url, files, suffix):
        corpora = {}
        for file in files:
            corpus = {}
            with open(get_file(file, base_url + file + suffix, cache_dir='/tmp/bible.cache/'), "rb") as webfile:
                for idx, row in enumerate(csv.reader(codecs.iterdecode(webfile, 'utf-8'))):
                    if idx > 0:
                        segs = BibleDataset.normalize(row[4]).split()
                        if len(segs) > MAX_SENTENCE_LENGTH:
                            segs = segs[:MAX_SENTENCE_LENGTH]
                        corpus[tuple(int(v) for v in row[:-1])] = segs
            corpora[file] = corpus

        keysets = [set(ks.keys()) for ks in corpora.values()]
        intersection = sorted(list(keysets[0].intersection(*keysets[1:])))

        return {file: [corpus[key] for key in intersection] for (file, corpus) in corpora.items()}

    def pad_sentence(self, sentence, length):
        return sentence + [self.mapping['<end>']] * (length - len(sentence))

    def sample_batch(self, batch_size):
        style_names = list(self.tokenized_corpora.keys())
        styles = random.choices(range(len(self.tokenized_corpora)), k=batch_size)
        sample = zip(styles, random.sample(range(len(list(self.tokenized_corpora.values())[0])), batch_size))
        return [self.pad_sentence(self.tokenized_corpora[style_names[corp]][sent], self.max_sentence_length)
                for corp, sent in sample], styles

    def enc_input(self, batch):
        return np.array([s + [self.mapping['<end>']] for s in batch], int)

    def dec_input(self, batch):
        return np.array([[self.mapping['<start>']] + s for s in batch], int)

    def gen_autoenc(self, batch_size=64, vocab_size=10000):
        while True:
            batch, _ = self.sample_batch(batch_size)
            dec_input = self.dec_input(batch)

            yield self.enc_input(batch), dec_input, keras.utils.to_categorical(dec_input, len(self.mapping)).astype(int)

    def gen_adversarial(self, batch_size=64, vocab_size=10000):
        while True:
            batch, styles = self.sample_batch(batch_size=batch_size)
            yield self.enc_input(batch), keras.utils.to_categorical(styles, len(self.tokenized_corpora)).astype(int)

    def gen_complete(self, batch_size=64, vocab_size=10000):
        while True:
            batch, styles = self.sample_batch(batch_size=batch_size)
            dec_input = self.dec_input(batch)
            yield self.enc_input(batch), dec_input, \
                  keras.utils.to_categorical(dec_input, len(self.mapping)).astype(int), \
                  keras.utils.to_categorical(styles, len(self.tokenized_corpora)).astype(int)

    @staticmethod
    def normalize(sentence):
        sentence = Dataset.normalize(sentence)

        psalms = re.findall('psalm [0-9]+', sentence)

        if len(psalms) > 0:
            sentence = sentence.split(psalms[0])[0]

        sentence = re.sub(r'\{(.*?)\}', '', sentence)
        sentence = re.sub('[0-9]+ <COLON> [0-9]+', '', sentence)

        return sentence


if __name__ == '__main__':
    c = BibleDataset(URL_ROOT, ["asv", "bbe", "dby", "kjv", "wbt", "web", "ylt"], CSV_EXT)

    c.gen_complete()
