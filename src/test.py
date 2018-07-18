from models import D_G_Model
from dataset.bible import Num2WordsDataset
import numpy as np

import keras
from keras.callbacks import ReduceLROnPlateau

batch_size = 6400


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = {'loss': [], 'val_loss': []}

    # def on_train_begin(self, logs={}):
    #  pass

    def on_epoch_end(self, batch, logs={}):
        for loss in ['loss', 'val_loss']:
            self.losses[loss].append(logs.get(loss))


np.random.seed(42)

import matplotlib.pyplot as plt


# summarize history for loss
def plt_losses(loss_history, title, with_val=False):
    plt.plot(loss_history.losses['loss'][:])
    if with_val:
        plt.plot(loss_history.losses['val_loss'][:])
    plt.title(title)
    med = 0.1 if len(loss_history.losses['loss']) < 1 else np.median(loss_history.losses['loss'])
    plt.ylim(ymin=-0.1)
    plt.ylim(ymax=med + 1.5)
    plt.ylabel('loss')
    plt.xlabel('batchse')
    plt.legend(['train', 'val'], loc='upper right')


def plt_all(with_val=True):
    plt.figure(figsize=(14, 4))
    plt.subplot(131)  # numrows, numcols, fignum
    plt_losses(loss_history, 'g loss', with_val)
    plt.subplot(132)
    plt_losses(loss_history_d, 'd loss', with_val)
    plt.subplot(133)
    plt_losses(loss_history_adv, 'adv loss', with_val)
    plt.show()


#
loss_history = LossHistory()
loss_history_d = LossHistory()
loss_history_adv = LossHistory()


# plt_all()


def train_g(g, steps, validation_steps=1):
    g.fit_generator(dataset.gen_g(dataset.train, batch_size),
                    steps,
                    validation_steps=validation_steps,
                    validation_data=dataset.gen_g(dataset.val, batch_size),
                    callbacks=[loss_history],
                    verbose=0 if steps < 10 else 1,
                    max_queue_size=50,
                    workers=2
                    )

    # model.fit([x_train, x_train_d], y_train,


def train_d(d, steps, validation_steps=1):
    # %time d_encoder_model.set_weights(encoder_model.get_weights())
    d.fit_generator(dataset.gen_d(dataset.train, batch_size),
                    steps,
                    validation_steps=validation_steps,
                    validation_data=dataset.gen_d(dataset.val, batch_size),
                    verbose=0 if steps < 20 else 1,
                    max_queue_size=50,
                    workers=2,
                    callbacks=[loss_history_d])
    # d.fit(x_train, style_train,


def train_adv(adv_model, steps, validation_steps=1):
    # %time classifier_model.set_weights(d_classifier_model.get_weights())

    adv_model.fit_generator(dataset.gen_adv(dataset.train, batch_size),
                            steps,
                            validation_steps=validation_steps,
                            validation_data=dataset.gen_adv(dataset.val, batch_size),
                            max_queue_size=50,
                            workers=2,
                            verbose=0 if steps < 20 else 1,
                            callbacks=[loss_history_adv])


dataset = Num2WordsDataset(start=1, end=1000000)

model = D_G_Model(num_encoder_tokens=len(dataset.word2index),
                  num_decoder_tokens=len(dataset.word2index),  # from dataset 3628
                  style_out_size=len(dataset.style2index),  # from dataset 2
                  cuddlstm=False,
                  latent_dim=512,  # twice the default. make it stronger! but slower
                  bidi_encoder=True,
                  adv_loss_weight=1.0, )
model.build_all()

from decoder import SamplingDecoder

sampler = SamplingDecoder(model)
sampler.show_sample(dataset, 'train', sample_ids=[0], teacher_forcing=True)

train_size = len(dataset.index2style) * 1000000
epoc = int(train_size/batch_size)
print('epoc is of',epoc,'of batches',batch_size,'total train_size',train_size)

for i in range(20):
    print('EPOC', i)
    train_g(model.g, int(epoc), validation_steps=int(epoc / 10))  # pretrain

    l = loss_history.losses['loss'][-1:][0]
    plt_all()
    sampler.show_sample(dataset, 'train', [0], teacher_forcing=False)
    sampler.show_sample(dataset, 'train', [0], teacher_forcing=True)
    if l < 0.5:  # 0.5 is good value for 20 length sentences. for longer, NOT!
        print('early break at break at', l, 'epoc', i)
        break

for i in range(0):
    train_d(model.d, epoc, validation_steps=100)
    l = loss_history_d.losses['loss'][-1:][0]
    if l < 2:
        print('early break at break at', l, 'epoc', i)
        break
