import random

import h5py
import numpy as np
from keras import Input, Model
from keras.layers import Dropout, Dense, Embedding, LSTM, add, BatchNormalization, TimeDistributed
import keras
import requests
from tcn import tcn

from src.dataset.bible import BibleDataset, URL_ROOT, CSV_EXT, START_SYMBOL, END_SYMBOL

EMBEDDINGS_GDRIVE_ID = '1w0GYh65l8r21IHTT6kVvdTMKIg6Y_i1M'

BATCH_SIZE = 64
NUM_EPOCHS = 100
EMBEDDING_D = 40
BEFORE_SOFTMAX_D = 40
LSTM_HIDDEN_DIMS = 64
BEFORE_CONCAT = EMBEDDING_D


class DataGenerator:
    def __init__(self, dataset, embeddings):
        self.dataset = dataset
        self.embeddings = embeddings
        pass

    def create_sequences(self, file, batch, max_sent_len=None, one_hot=True):
        X1, X2, y = list(), list(), list()
        # walk through each sentence in batch
        max_size = max_sent_len if max_sent_len else self.dataset.max_sentence_length + 2
        for num in batch:
            seq = [self.dataset.word2index[START_SYMBOL]] + self.dataset.corpora[file][num]
            # split one sequence into multiple X,y pairs
            # for i in range(1, len(seq)):
            i = len(seq)
            # split into input and output pair
            in_seq, out_seq = seq[:], seq[1:]
            # pad input sequence

            in_seq = self.dataset.pad_sentence(in_seq, max_size)
            out_seq = self.dataset.pad_sentence(out_seq, max_size)

            # encode output sequence
            if one_hot:
                out_seq = keras.utils.to_categorical(out_seq, num_classes=len(self.dataset.word2index))

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

    def generate(self, file, data, batch_size, max_sent_len=None, one_hot=True):
        """ max_sent_len should be below model allowed size."""
        while True:
            batch = random.sample(range(*data), k=min(batch_size, len(range(*data))))
            yield self.create_sequences(file, batch, max_sent_len, one_hot)


def decorate_file(file):
    return '<{}>'.format(file)


def load_embeddings(emb_path, files):
    with h5py.File(emb_path, "r") as ds:
        return {decorate_file(file): np.array(ds[file]) for file in files if file in ds}


def define_model(dataset):
    semantic_dropout = 0.1
    lstm_in_dropout = 0.1
    final_dropout = 0.1

    # Sentence Embeddings
    max_length = dataset.max_sentence_length + 2
    vocab_size = len(dataset.word2index)
    print('max_length', max_length, 'vocab_size', vocab_size)

    inputs1 = Input(shape=(4096,), name='f_input')  # means BATCHx4096
    fe1 = Dropout(0.25, name='fe1')(inputs1)
    fe2 = Dense(BEFORE_CONCAT, name='fe2', activation='relu')(fe1)

    # sequence model.
    # Note that shape is of the sample (ignoring batch d).  (50, 4096) (50, 62) (50, 62, 3626)
    inputs2 = Input(shape=(None,), name='s_input')  # TPDP: remove dim here!!!
    se1 = Embedding(vocab_size, EMBEDDING_D, mask_zero=False, name='se1')(inputs2)  # why mask_Zero=True
    print("false to masking")
    # se2 = Dropout(lstm_in_dropout,name='se2')(se1)
    # benchmark adapted from Penn treebank k=3. n=4. hidden=600 dropout=0.5
    se = tcn.TCN(input_layer=se1, nb_filters=BEFORE_CONCAT, kernel_size=3, nb_stacks=2, dilations=[1, 2, 4, 8],
                 dropout_rate=0.2, return_sequences=True)
    # se3 = TimeDistributed(Dense(BEFORE_CONCAT,name='se4'))(se2)

    msk1 = keras.layers.Masking(mask_value=0.0)(se1)
    msk1 = keras.layers.Lambda(lambda x: x * 0.0)(msk1)

    # decoder model
    decoder1 = add([fe2, se, msk1], name='concat/add')
    decoder2 = TimeDistributed(Dense(BEFORE_SOFTMAX_D, activation='relu', name='decoder2'))(decoder1)
    decoder2 = TimeDistributed(Dropout(final_dropout, name='final_dropout'))(decoder2)
    outputs = TimeDistributed(Dense(vocab_size, name='outputs', activation='softmax'))(decoder2)  # activation='softmax

    # tie it together [image, seq] [wordo
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.summary()
    return model


def train_model(dataset, embeddings):
    generator = DataGenerator(dataset, embeddings)
    model = define_model(dataset)

    optimizer = keras.optimizers.Adam(lr=0.002, clipnorm=0.4)  # Grad Clip was 0.4 on aricle, lr 0.002
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    train_generator = generator.generate("<ylt>", dataset.train, BATCH_SIZE, one_hot=False)
    val_generator = generator.generate("<ylt>", dataset.val, BATCH_SIZE, one_hot=False)
    i = 0
    cp_callback = keras.callbacks.ModelCheckpoint("/tmp/model-{epoch:03d}-{val_loss:.2f}.hdf5", monitor='val_acc',
                                                  save_best_only=True, save_weights_only=False)
    tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/Graph', histogram_freq=0, write_graph=True,
                                              write_images=True)
    for i in range(1000):
        model.fit_generator(train_generator,
                            steps_per_epoch=int(dataset.train[1] / BATCH_SIZE),
                            epochs=100,
                            validation_data=val_generator,
                            validation_steps=2,
                            verbose=True)


def test(model, dataset, embeddings, sent=100, jump=1):
    for snum in range(0, sent * jump, jump):
        in_text = [dataset.word2index[START_SYMBOL]]
        for i in range(dataset.max_sentence_length + 2):
            # pad input
            sequence = dataset.pad_sentence(in_text, dataset.max_sentence_length + 2)
            # predict next word-seq. in practice we only take the i-th one
            yhat = model.predict([embeddings['<bbe>'][snum].reshape(1, -1), np.array(sequence).reshape(1, -1)],
                                 verbose=0)
            # print (yhat.shape) #1x62x3626
            # convert probability to integer
            SENT_IN_BATCH = 0
            yhat = np.argmax(yhat[SENT_IN_BATCH][i])  # FIX THIS:
            # print (yhat, np.array(sequence).reshape(1, -1).shape,sequence)
            # map integer to word
            in_text += [yhat]
            # stop if we predict the end of the sequence
            if yhat == dataset.word2index[END_SYMBOL]:
                break
        # print('{}({}): {}'.format(snum, file, dataset.recostruct_sentence(in_text)))
        reference = [dataset.recostruct_sentence(dataset.corpora['<ylt>'][snum])]
        sentence = dataset.recostruct_sentence(in_text[1:-1])
        from nltk.translate.bleu_score import sentence_bleu
        score = sentence_bleu(reference, sentence, weights=(0.5, 0.5, 0, 0.0))
        print('{} (score: {}): P: "{}" G: "{}"'.format(snum, score, sentence, reference))


def emit_predictions(model, dataset, embeddings, rng):
    sentences = []
    for snum in rng:
        in_text = [dataset.word2index[START_SYMBOL]]
        for i in range(dataset.max_sentence_length + 2):
            # pad input
            sequence = dataset.pad_sentence(in_text, dataset.max_sentence_length + 2)
            # predict next word-seq. in practice we only take the i-th one
            yhat = model.predict([embeddings['<bbe>'][snum].reshape(1, -1), np.array(sequence).reshape(1, -1)],
                                 verbose=0)
            # print (yhat.shape) #1x62x3626
            # convert probability to integer
            SENT_IN_BATCH = 0
            yhat = np.argmax(yhat[SENT_IN_BATCH][i])  # FIX THIS:
            # print (yhat, np.array(sequence).reshape(1, -1).shape,sequence)
            # map integer to word
            in_text += [yhat]
            # stop if we predict the end of the sequence
            if yhat == dataset.word2index[END_SYMBOL]:
                break
        # print('{}({}): {}'.format(snum, file, dataset.recostruct_sentence(in_text)))
        sentences += [[dataset.index2word[s] for s in in_text[1:-1]]]
    return sentences


def emit_reference(dataset, rng):
    sentences = []
    for snum in rng:
        sentences += [[dataset.index2word[s] for s in dataset.corpora['<ylt>'][snum]]]

    return sentences


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def dload_embeddings(path, files):
    download_file_from_google_drive(EMBEDDINGS_GDRIVE_ID, path)
    return load_embeddings(path, files)


def calculate_bleu_score(lstm_path, tcnn_path, emb_path, samples):
    model = keras.models.load_model(lstm_path)
    dataset = BibleDataset(["bbe", "ylt"])
    embeddings = load_embeddings(emb_path, ["bbe", "ylt"])
    rng = range(len(dataset.corpora['<ylt>']))
    if samples > 0:
        rng = random.sample(list(rng), samples)
    h = emit_predictions(model, dataset, embeddings, rng)
    r = [[s] for s in emit_reference(dataset, rng)]
    from nltk.translate.bleu_score import corpus_bleu
    score1 = corpus_bleu(r, h, (1, 0, 0, 0))
    score2 = corpus_bleu(r, h, (0.5, 0.5, 0, 0))
    score3 = corpus_bleu(r, h, (0.33, 0.33, 0.33, 0))
    score4 = corpus_bleu(r, h)
    print("LSTM: {}".format([score1, score2, score3, score4]))
    model = keras.models.load_model(tcnn_path)
    h = emit_predictions(model, dataset, embeddings, rng)
    score1 = corpus_bleu(r, h, (1, 0, 0, 0))
    score2 = corpus_bleu(r, h, (0.5, 0.5, 0, 0))
    score3 = corpus_bleu(r, h, (0.33, 0.33, 0.33, 0))
    score4 = corpus_bleu(r, h)
    print("TCNN: {}".format([score1, score2, score3, score4]))


if __name__ == '__main__':
    calculate_bleu_score("./exp/uni_embed/lstm/model.h5", "./exp/uni_embed/tcnn/model.h5", "/tmp/emb.h5", 200)
