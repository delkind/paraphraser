import codecs
import csv
import random
import re
from collections import Counter
from nltk.tokenize import word_tokenize

import numpy as np
from keras.utils import to_categorical
from keras.utils.data_utils import get_file

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
        return END_SYMBOL

    def create_mapping(self, corpora):
        segs = [seg for corpus in corpora.values() for sentence in corpus for seg in sentence]
        frequencies = Counter(segs).items()
        oov = [c for (_, c) in frequencies if c <= MIN_FREQ]
        frequencies = {k: l for (k, l) in frequencies if l > MIN_FREQ or k == OOV}
        frequencies[OOV] = sum(oov)
        markers = [END_SYMBOL, START_SYMBOL] + list(corpora.keys())
        self.word2index = {seg: num for (num, seg) in enumerate(markers + sorted(list(frequencies.keys())))}
        # for m in sorted(markers):
        #    self.word2index[m] = len(self.word2index)
        assert (self.word2index[END_SYMBOL] == 0)  # in embedding_layer, mask_true assumes padding=0
        self.index2word = {v: k for k, v in self.word2index.items()}

    def index(self, corpora):
        self.create_mapping(corpora)
        oov = self.word2index[OOV]
        return {key: [[self.word2index.get(seg, oov) for seg in sentence] for sentence in corpus]
                for (key, corpus) in corpora.items()}

    def normalize(self, sentence):
        sentence = re.sub('[^ a-zA-Z0-9.,:;!\?\'{}]', ' ', sentence.lower())
        return sentence.strip()

    def build_embeddings(self, sentences, w2v_path):
        word_dict = self.get_word_dict(sentences)
        return self.get_w2v(word_dict, w2v_path)

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.word2index[START_SYMBOL]] = ''
        word_dict[self.word2index[END_SYMBOL]] = ''
        return word_dict

    def get_w2v(self, word_dict, w2v_path):
        # create word_vec with w2v vectors
        word_vec = {}
        with open(w2v_path, encoding="utf-8") as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in self.word2index:
                    if self.word2index[word] in word_dict:
                        word_vec[word] = np.fromstring(vec, sep=' ')
                    else:
                        print("Embedding not found for " + self.word2index[word])
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec


def decorate_file(file):
    return '<{}>'.format(file)


class BibleDataset(Dataset):
    def __init__(self, files, base_url=URL_ROOT, suffix=CSV_EXT, test_split=0.1, validation_split=0.1,
                 v2w_path=None):
        super().__init__()
        corpora, index = self.parse_csv(base_url, files, suffix)
        self.corpora = self.index(corpora)
        self.train, self.val, self.test = self.split(index, test_split, validation_split)
        sentence_lengths = [len(s) for c in self.corpora.values() for s in c]
        self.max_sentence_length = max(sentence_lengths)
        self.style2index = {s: i for (i, s) in enumerate(corpora.keys())}
        self.index2style = {v: k for k, v in self.style2index.items()}
        if v2w_path is not None:
            sentences = []
            for file in files:
                sentences += self.corpora[decorate_file(file)]
            self.word_vec = self.build_embeddings(sentences, v2w_path)
        pass

    def parse_csv(self, base_url, files, suffix):
        corpora = {}
        for file in files:
            corpus = {}
            with open(get_file(file, base_url + file + suffix, cache_dir='/tmp/bible.cache/'), "rb") as webfile:
                for idx, row in enumerate(csv.reader(codecs.iterdecode(webfile, 'utf-8'))):
                    if idx > 0:
                        segs = [str(s).strip() for s in word_tokenize(self.normalize(row[4]))
                                if len(str(s).strip()) > 0]

                        if len(segs) > MAX_SENTENCE_LENGTH:
                            segs = segs[:MAX_SENTENCE_LENGTH]
                        corpus[tuple(int(v) for v in row[:-1])] = segs
            corpora[decorate_file(file)] = corpus

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
        # note length can be small, creating truncation, len(sentence)=60 but length=15
        return sentence[:length] + [self.word2index[END_SYMBOL]] * max(0, length - len(sentence))

    def recostruct_sentence(self, sentence):
        return ' '.join([self.index2word[seg] for seg in sentence])

    def create_sequences(self, file, batch, max_sent_len=None, one_hot=True):
        X1, X2, y = list(), list(), list()
        # walk through each sentence in batch
        max_size = max_sent_len if max_sent_len else self.max_sentence_length + 2
        for num in batch:
            seq = [self.word2index[START_SYMBOL]] + self.corpora[file][num]
            # split one sequence into multiple X,y pairs
            # for i in range(1, len(seq)):
            i = len(seq)
            # split into input and output pair
            in_seq, out_seq = seq[:], seq[1:]
            # pad input sequence

            in_seq = self.pad_sentence(in_seq, max_size)
            out_seq = self.pad_sentence(out_seq, max_size)

            # encode output sequence
            if one_hot:
                out_seq = to_categorical(out_seq, num_classes=len(self.word2index))

            # print (len(in_seq),len(out_seq),out_seq.shape)
            # store
            X1.append(self.embeddings[file][num])
            X2.append(in_seq)
            y.append(out_seq)
        if one_hot:
            batch_y = np.array(y, dtype=np.int8)
        else:
            batch_y = np.array(y, dtype=np.int32)[:, :, np.newaxis]  # create last axis as 1
        return [[np.array(X1), np.array(X2)], batch_y]

    def data_generator(self, file, data, batch_size, max_sent_len=None, one_hot=True):
        """ max_sent_len should be below model allowed size."""
        while True:
            batch = random.sample(range(*data), k=min(batch_size, len(range(*data))))
            yield self.create_sequences(file, batch, max_sent_len, one_hot)

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
