import num2words
from dataset.bible import *


class NumSentenceGen(object):
    def __init__(self, rng, indexer,use_num_names):
        """

        :param rng:
        :param indexer:
        :param use_num_names: instead of returing '1 2' return 'one two'
        """
        self.range = rng
        self.indexer = indexer
        self.word_producer = num2words.lang_EN.Num2Word_EN()
        self.use_num_names = use_num_names

    def __getitem__(self, item):
        if type(item) == int:
            return self.indexer(self.__get_sentence__(item))
        elif type(item) == slice:
            return [self[i] for i in range(item.start, item.stop)]
        else:
            return [self[i] for i in item]

    def __len__(self):
        return len(self.range)

    def __get_sentence__(self, number):
        digits = list(str(number))

        if self.use_num_names:
            return  [self.word_producer.to_cardinal(int(digit)) for digit in digits]
        else:
            return digits

class NumWordedSentenceGen(NumSentenceGen):
    def __init__(self, rng, indexer, normalizer):
        super().__init__(rng, indexer,False)
        self.normalizer = normalizer
        self.word_producer = num2words.lang_EN.Num2Word_EN()

    def __get_sentence__(self, number):
        return self.normalizer(self.word_producer.to_cardinal(number)).split()


class Num2WordsDataset(Dataset):
    def __init__(self, start=1, end=1000000, test_split=0.1, validation_split=0.1,use_num_names=False):
        super().__init__()
        self.range = range(start, end)
        indexer = lambda s: self.sentence2indexes(s)
        self.corpora = {'<num>': NumSentenceGen(range(start, end), indexer=indexer,use_num_names= use_num_names),
                        '<wrd>': NumWordedSentenceGen(range(start, end), indexer=indexer,
                                                      normalizer=lambda s: self.normalize(s))}
        numbers = list('0123456789')
        self.word2index = {k: v for v, k in enumerate(sorted([card for card in self.corpora['<wrd>'].word_producer.
                                                             cards.values()] + self.corpora['<wrd>'].word_producer.
                                                             exclude_title + list(self.corpora.keys()) +
                                                             numbers + [END_SYMBOL]))}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.index2style = {k: v for k, v in enumerate(list(self.corpora.keys()))}
        self.style2index = {v: k for k, v in self.index2style.items()}

        val_count = int(len(self.range) * validation_split)
        test_count = int(len(self.range) * test_split)
        train_count = len(self.range) - val_count - test_count

        self.val = (start, val_count + 1)
        self.test = (self.val[1], self.val[1] + test_count)
        self.train = (self.test[1], self.val[1] + train_count)

        longest_num = int('9' * len(str(end - 1)))

        self.max_sentence_length = max([len(corpus[longest_num]) for corpus in self.corpora.values()])





    def sentence2indexes(self, sentence):
        return [self.word2index[seg] for seg in sentence]

    def pad_sentence(self, sentence, length):
        return sentence + [self.word2index[END_SYMBOL]] * (length - len(sentence))

    def recostruct_sentence(self, sentence):
        return ' '.join([self.index2word[seg] for seg in sentence])

    def sample_batch(self, data, batch_size, ):
        sample = list(zip(random.choices(list(self.corpora.keys()), k=batch_size),
                          random.sample(range(*data), k=batch_size)))

        max_len = max([len(self.corpora[style][sent]) for style, sent in sample])
        return [self.pad_sentence(self.corpora[style][sent], max_len) for style, sent in
                sample], [self.style2index[style] for style, _ in sample]

    def noisy_sample_batch(self, data, batch_size, one_style_only=None, noise_std=1.0):
        """ noise is np.normal using that std (applying floor use 0.0 for no noise.
         use noise_std of 1.0 if you want  67% to be 0, and some 1 and rarely 2
        """
        if one_style_only:
            styles = [one_style_only] *batch_size
        else:
            styles = random.choices(list(self.corpora.keys()), k=batch_size)

        sample = list(zip(styles, random.sample(range(*data), k=batch_size)))

        normal_noise = np.random.normal(0,noise_std,len(sample))
        noise_sample=[]
        for i,(style,num) in enumerate(sample):
            noisy_num = min(max(data[0],num+ int(normal_noise[i])),data[1]) #must be range
            #print (num,noisy_num)
            noise_sample.append( (style,noisy_num) )


        max_len = max(max([len(self.corpora[style][sent]) for style, sent in sample])
                     ,max([len(self.corpora[style][sent]) for style, sent in noise_sample]))
        results = []
        for sample in [sample,noise_sample]:
            res=[self.pad_sentence(self.corpora[style][sent], max_len) for style, sent in
                    sample], [self.style2index[style] for style, _ in sample]
            results.append(res)

        return results


    def enc_input(self, batch):
        return np.array([s + [self.word2index[END_SYMBOL]] for s in batch], int)

    def dec_input(self, batch, styles):
        return np.array([[self.word2index[style]] + s for (s, style) in zip(batch, styles)], int)





    def gen_g(self, data_range, batch_size=64,noise_std=0.0):
        assert type(batch_size)==int
        while True:
            #batch, styles = self.sample_batch(data_range, batch_size=batch_size)
            (batch, styles),(batch_noise,styles_noise) = self.noisy_sample_batch(data_range, batch_size=batch_size, noise_std=noise_std)

            dec_input = self.dec_input(batch, [self.index2style[style] for style in styles])
            enc_input = self.enc_input(batch_noise)
            #print (batch) #end10 style11,12
            yield [enc_input, dec_input], to_categorical(enc_input, len(self.word2index)).astype(int)

    def gen_d(self, data_range, batch_size=64, noise=0.0):
        """
        :param data_range:
        :param batch_size:
        :param noise: delibratly switch label value of noise*batch_size
        :return:
        """
        while True:
            batch, styles = self.sample_batch(data_range, batch_size=batch_size)

            if int(noise*batch_size)>0:
                noise_ind = np.random.choice(range(batch_size), int(batch_size * noise), replace=False)
                styles = np.array(styles)
                styles[noise_ind]  = np.random.random_integers(0,len(self.style2index)-1,len(noise_ind))
                #print (styles.shape,styles)
            # ONLY SUPPORT 1 OR 0
            yield self.enc_input(batch), styles #to_categorical(styles, len(self.corpora)).astype(int)

    def gen_adv(self, data_range, batch_size=64, style_noise=0.0, noise_std=0.0):
        """
        :param data_range:
        :param batch_size:
        :param style_noise: change style label . what probability to gflip (0-1) default 0
        :param noise_std: noise on the input number. draw from normal dist. with noise_std (24 will be 25 if 1 drawn)
        :return: tuple : (enc_input, dec_input) (one-hot enc_input ,  styles)
        """
        while True:
            #batch, styles = self.sample_batch(data_range, batch_size=batch_size)
            (batch, styles), (batch_noise, styles_noise) = self.noisy_sample_batch(data_range, batch_size=batch_size,
                                                                                   noise_std=noise_std)
            enc_input = self.enc_input(batch_noise)
            dec_input = self.dec_input(batch, [self.index2style[style] for style in styles])
            #print(len(styles),len(batch))
            if int(style_noise*batch_size)>0:
                noise_ind = np.random.choice(range(batch_size), int(batch_size * style_noise), replace=False)
                styles = np.array(styles)
                #print(styles.shape,styles)
                styles[noise_ind]  = np.random.random_integers(0,len(self.style2index)-1,len(noise_ind))
            #print (enc_input.shape)
            yield [enc_input, dec_input], \
                  [to_categorical(enc_input, len(self.word2index)).astype(int),
                   np.array(styles) #to_categorical(styles, len(self.corpora)).astype(int)
                   ]

    def normalize(self, sentence):
        sentence = re.sub('[^ a-z]', ' ', sentence.lower())
        return sentence.strip()


def test1():
    # dataset = BibleDataset(URL_ROOT, ["asv", "bbe", "dby", "kjv", "wbt", "web", "ylt"], CSV_EXT)
    dataset = Num2WordsDataset()
    print(next(dataset.gen_adv(dataset.train)))
    pass
    print('loading dataset')
    dataset = BibleDataset(["bbe", "ylt"])

    for i in range(1):
        # WORKS ON TRAIN g_adv = dataset.gen_adv(dataset.train, batch_size)
        g_adv = dataset.gen_adv(dataset.val, 64)
        print('staring to create 100 batches. if it is slow, try to imporve the code!')
        for _ in range(100):
            next(g_adv)
        print('end of creating 100 batches')

    print(dataset.style2index)
    print(list(dataset.word2index.items())[-6:])
    [x1, x2], [y1, y2] = next(dataset.gen_adv(dataset.val, 64))
    assert np.allclose(np.argmax(y1[0][:10], axis=1), x1[0][:10])

    [x1, x2], y1 = next(dataset.gen_g(dataset.val, 64))
    assert np.allclose(np.argmax(y1[0][:10], axis=1), x1[0][:10])

def test2():
    for dataset in [Num2WordsDataset(end=5000), Num2WordsDataset(end=5000,use_num_names=True)]:
        print (dataset.max_sentence_length) #one hundred nighty nine
        #next(dataset.gen_d(dataset.train,10,noise=0.4))
        (x1,x2),y1=next(dataset.gen_g(dataset.train, batch_size=5, noise_std=2       ))
        #(x1, x2), (y1,y2) = next(dataset.gen_adv(dataset.train, 100, noise_std=0))
        print (dataset.recostruct_sentence(x1[0]))
        print (dataset.recostruct_sentence(x2[0]))




if __name__ == '__main__':
    test2()
