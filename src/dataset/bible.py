import codecs
import csv
import math
import random
import re
from collections import Counter

import numpy as np
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from itertools import product
import num2words

END_SYMBOL = '<end>'
START_SYMBOL = '<start>'
# SPEC_CHARS_REPLACEMENT = {'.': ' <DOT> ', ',': ' <COMMA> ', ';': ' <SEMICOLON> ', ':': ' <COLON> ', '!': ' <EXCL> ',
#                           '?': ' <QSTN> ', "'": ' <QUOTE> '}
# SPEC_CHARS_RECONSTRUCT = {' ' + v.strip(): k for (k, v) in SPEC_CHARS_REPLACEMENT.items()}

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


class Dataset:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}

    @staticmethod
    def end_symbol():
        """ as string """
        return END_SYMBOL

    def create_mapping(self, corpora):
        segs = [seg for corpus in corpora.values() for sentence in corpus for seg in sentence]
        frequencies = Counter(segs).items()
        oov = [c for (_, c) in frequencies if c <= MIN_FREQ]
        frequencies = {k: l for (k, l) in frequencies if l > MIN_FREQ or k == OOV}
        frequencies[OOV] = sum(oov)
        self.word2index = {seg: num for (num, seg) in enumerate(sorted(list(frequencies.keys())))}
        markers = [START_SYMBOL, END_SYMBOL] + list(corpora.keys())
        for m in sorted(markers):
            self.word2index[m] = len(self.word2index)
        self.index2word = {v: k for k, v in self.word2index.items()}

    def index(self, corpora):
        self.create_mapping(corpora)
        oov = self.word2index[OOV]
        return {key: [[self.word2index.get(seg, oov) for seg in sentence] for sentence in corpus]
                for (key, corpus) in corpora.items()}

    def normalize(self, sentence):
        sentence = re.sub('[^ a-zA-Z0-9.,:;!\?\'{}]', ' ', sentence.lower())
        return sentence.strip()


class BibleDataset(Dataset):
    def __init__(self, files, base_url=URL_ROOT, suffix=CSV_EXT, test_split=0.1, validation_split=0.1):
        super().__init__()
        corpora, index = self.parse_csv(base_url, files, suffix)
        self.corpora = self.index(corpora)
        self.train, self.val, self.test = self.split(index, test_split, validation_split)
        sentence_lengths = [len(s) for c in self.corpora.values() for s in c]
        self.max_sentence_length = max(sentence_lengths)
        self.style2index = {s: i for (i, s) in enumerate(corpora.keys())}
        self.index2style = {v: k for k, v in self.style2index.items()}
        self.clusters = self.cluster(10)
        pass

    def parse_csv(self, base_url, files, suffix):
        from spacy.lang.en import English
        tokenizer = English().Defaults.create_tokenizer()
        corpora = {}
        for file in files:
            corpus = {}
            with open(get_file(file, base_url + file + suffix, cache_dir='/tmp/bible.cache/'), "rb") as webfile:
                for idx, row in enumerate(csv.reader(codecs.iterdecode(webfile, 'utf-8'))):
                    if idx > 0:
                        segs = [str(s).strip() for s in tokenizer(self.normalize(row[4]))
                                if len(str(s).strip()) > 0]

                        if len(segs) > MAX_SENTENCE_LENGTH:
                            segs = segs[:MAX_SENTENCE_LENGTH]
                        corpus[tuple(int(v) for v in row[:-1])] = segs
            corpora['<{}>'.format(file)] = corpus

        keysets = [set(ks.keys()) for ks in corpora.values()]
        intersection = [s for s in sorted(list(keysets[0].intersection(*keysets[1:])))
                        if len([corp[s] for corp in corpora.values() if len(corp[s]) == 0]) == 0]

        corpora = {file: [corpus[key] for key in intersection] for (file, corpus) in corpora.items()}
        index = [key[1] for key in intersection]

        return corpora, index

    @staticmethod
    def split(index, test_split, validation_split):
        count_test = len(index) * test_split
        count_validation = len(index) * validation_split
        count = Counter(index)

        sum_validation = 0
        sum_test = 0

        test = []
        validation = []

        for i in reversed(range(1, len(count) + 1)):
            if sum_test < count_test:
                sum_test += count[i]
                test += [i]
            elif sum_validation < count_validation:
                sum_validation += count[i]
                validation += [i]
            else:
                break
        test = set(test)
        validation = set(validation)
        train = set(index) - test - validation

        train = (0, sum([count[i] for i in train]))
        validation = (train[1], train[1] + sum([count[i] for i in validation]))
        test = (validation[1], validation[1] + sum([count[i] for i in test]))

        return train, test, validation

    def pad_sentence(self, sentence, length):
        """ sentence is a list . length is max length of sentence"""
        return sentence + [self.word2index[END_SYMBOL]] * (length - len(sentence))

    def recostruct_sentence(self, sentence):
        return ' '.join([self.index2word[seg] for seg in sentence])

    def sample_batch(self, data, batch_size, max_len):
        sample = random.sample(data, k=min(batch_size, len(data)))
        return [self.pad_sentence(self.corpora[self.index2style[style]][sent], max_len)
                for style, sent in sample], [style for style, _ in sample]






    def enc_input(self, batch):
        """ batch is a list of senteces(list of tokens)
        will add <end> symbol to each of the sentences and return it as np.array of type int
        """
        return np.array([s + [self.word2index[END_SYMBOL]] for s in batch], int)

    def dec_input(self, batch, styles):
        return np.array([[self.word2index[style]] + s for (s, style) in zip(batch, styles)], int)

    def gen_pair(self, data_range, batch_size=64):
        cluster, data = self.__generate_data__(data_range)
        max_len = self.clusters[cluster][0]
        while True:
            if len(data)<batch_size:
                raise ValueError('fbatch size {batch_size}smaller than len of data{len(data}')

            same_ids = random.sample(data, k=batch_size//2) # style,sent (the first ignored)
            not_same_batch = (batch_size+1)//2

            not_same_ids = random.sample(data, 2*not_same_batch)  # style,sent (the first ignored)

            # half the pair same id, different styles
            # other half, different ids, random styles?
            batch0 = self.enc_input([self.pad_sentence(self.corpora['<bbe>'][sent], max_len) for _, sent in sample_ids])
            batch1 = self.enc_input([self.pad_sentence(self.corpora['<ylt>'][sent], max_len) for _, sent in sample_ids])

            dec_input = self.dec_input(batch, [self.index2style[style] for style in styles])
            enc_input = self.enc_input(batch)
            yield [enc_input, dec_input], to_categorical(enc_input, len(self.word2index)).astype(int)



    def gen_g(self, data_range, batch_size=64):
        cluster, data = self.__generate_data__(data_range)
        while True:
            batch, styles = self.sample_batch(data, batch_size=batch_size, max_len=self.clusters[cluster][0])
            dec_input = self.dec_input(batch, [self.index2style[style] for style in styles])
            enc_input = self.enc_input(batch)
            yield [enc_input, dec_input], to_categorical(enc_input, len(self.word2index)).astype(int)

    def gen_d(self, data_range, batch_size=64):
        cluster, data = self.__generate_data__(data_range)
        while True:
            batch, styles = self.sample_batch(data, batch_size=batch_size, max_len=self.clusters[cluster][0])
            yield self.enc_input(batch), to_categorical(styles, len(self.corpora)).astype(int)

    def gen_adv(self, data_range, batch_size=64):
        cluster, data = self.__generate_data__(data_range)
        while True:
            batch, styles = self.sample_batch(data, batch_size=batch_size, max_len=self.clusters[cluster][0])
            enc_input = self.enc_input(batch)
            dec_input = self.dec_input(batch, [self.index2style[style] for style in styles])
            yield [enc_input, dec_input], \
                  [to_categorical(enc_input, len(self.word2index)).astype(int),
                   to_categorical(styles, len(self.corpora)).astype(int)]

    def __generate_data__(self, data_range):
        cluster = np.random.choice(list(self.clusters.keys()), 1,
                                   p=[len(v[1]) / sum([len(v[1]) for v in self.clusters.values()])
                                      for v in self.clusters.values()])[0]
        data = list(set(product(self.index2style.keys(), range(*data_range))).intersection(
            self.clusters[cluster][1]))
        return cluster, data

    def normalize(self, sentence):
        sentence = super().normalize(sentence)
        psalms = re.findall('psalm [0-9]+', sentence)
        if len(psalms) > 0:
            sentence = sentence.split(psalms[0])[0]
        sentence = re.sub(r'\{(.*?)\}', '', sentence)
        sentence = re.sub('[0-9]+ <COLON> [0-9]+', '', sentence)

        return sentence

    def cluster(self, iteration):
        clusters = {i: [] for i in range(self.max_sentence_length // iteration + 1)}
        for style_index, style in self.index2style.items():
            for sent_index, sent in enumerate(self.corpora[style]):
                cluster = len(sent) // iteration
                clusters[cluster].append((style_index, sent_index))

        clusters = {k: (max([len(self.corpora[self.index2style[s[0]]][s[1]]) for s in v]), v) for k, v in
                    clusters.items()}

        return clusters







if __name__ == '__main__':
    pass
    # test_bible_dataset()

