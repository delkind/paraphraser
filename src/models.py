import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding,CuDNNLSTM,Bidirectional,Concatenate,Dropout
from keras import backend as K
import matplotlib.pyplot as plt


class D_G_Model:
    """
    # Model defintion
    (1) We start with regular seq2seq is encoder->embedding->decoder and trained with reconstruction-loss

    (2)We can build style-discriminator where the target is to classify author-style from the sentence-embedding.
    When training it you need to freeze the encoder and decoder parts of the model, then:
    input1: sentence --freezed encoder--> embedding    (no need to run decoder)
    input2: style (one-hot)
    output: style (one-hot)
    The discriminator can be a simple classifier (dense-based) with simple minimize cross-entropy target.

    (3) The smart-part: We want to train the encoder to create an embedding which will fool the discriminator.
    We will freeze the discriminator weights, and train the encoder-decoder similiarly to (1) with extra objective.
    That the loss from the discriminator will be Maximized.
    """

    def __init__(self, num_encoder_tokens,
                 num_decoder_tokens,  # from dataset 3628
                 style_out_size,  # from dataset 2
                 cuddlstm,
                 embedding_dim=300,  # 100-300 good numbers
                 # hidden unit of LSTM. Increasing it increase the model strength (and compuation time).
                 # increasing should increase fit to train (and when using stronger reguralization/dropout even val)
                 latent_dim=256,
                 bidi_encoder=False,
                 # LSTM dropouts, one of the best ways to fight overfiltting. ragnge 0-0.5++ (0 means no dropout)
                 # Increase it to reduce diff between train and val
                 en_lstm_dropout=0.3,
                 en_lstm_recurrent_dropout=0.3,  # only relevant if NOT cuddlstm.
                 de_lstm_dropout=0.0,  # only relevant if NOT cuddlstm.
                 de_lstm_recurent_dropout=0.0,  # only relevant if NOT cuddlstm
                 # how much weight to assign to adv-classifier compared to 1.0 of reconstruction-loss
                 # 1.0 - 100.0 means same. 10.0 means ten time more, etc
                 adv_loss_weight=1.0,
                 ):

        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.bidi_encoder = bidi_encoder
        self.cuddlstm = cuddlstm
        self.en_lstm_dropout = en_lstm_dropout
        self.en_lstm_recurrent_dropout = en_lstm_recurrent_dropout
        self.de_lstm_dropout = de_lstm_dropout
        self.de_lstm_recurent_dropout = de_lstm_recurent_dropout
        self.adv_loss_weight = adv_loss_weight
        self.style_out_size = style_out_size

        self.decoder_latent_dim = latent_dim * 2 if bidi_encoder else latent_dim
        self.shared_embedding = Embedding(num_encoder_tokens,
                                          embedding_dim,
                                          # weights=[word_embedding_matrix], if there is one (word2vec)
                                          # trainable=False,
                                          # input_length=MAX_SEQUENCE_LENGTH, if there is one
                                          )

    def build_encoder_model(self, encoder_inputs):
        # Dropout(noise_shape=(batch_size, 1, features))
        x = self.shared_embedding(encoder_inputs)
        if (self.cuddlstm):
            encoder_lstm = CuDNNLSTM(self.latent_dim, return_state=True)
        else:
            print('using LSTM with dropout!')
            # need to tune the dropout values (maybe fast.ai tips) , just invented those value
            # see dropout disucssion: https://github.com/keras-team/keras/issues/7290. iliaschalkidis
            encoder_lstm = LSTM(self.latent_dim, return_state=True, dropout=self.en_lstm_dropout,
                                recurrent_dropout=self.en_lstm_recurrent_dropout, name='rnn_encoder')
        if (self.bidi_encoder):
            encoder_lstm = Bidirectional(encoder_lstm, merge_mode='concat', name='rnn_encoder')
            x, forward_h, forward_c, backward_h, backward_c = encoder_lstm(x)  # output,h1,c1,h2,c2
            state_h = Concatenate()([forward_h, backward_h])
            state_c = Concatenate()([forward_c, backward_c])
        else:
            x, state_h, state_c = encoder_lstm(x)

        encoder_states = [state_h, state_c]  # sentence embedding LSTM: h,c
        encoder_model = Model(encoder_inputs, encoder_states)
        return encoder_model

    def build_decoder_model(self, encoder_model, encoder_inputs, decoder_inputs):
        # Set up the decoder, using `encoder_states` as initial state.
        # looks similiar to
        # gold decoder_ouputs: [1,0,0] [0,1,0] one-hot of the below
        # encoder_inputs:      hello   world <end> <end>
        # decoder_input      : <style> hello world <end>   | style = bible1,bible2,(maybe-generic for pertraining)
        # encoder_inputs = Input(shape=(None,),name='encoder_inputs')
        # decoder_inputs = Input(shape=(None,),name='decoder_inputs')

        # bi-di pass merge of h1+h2, c1+c2
        if (self.cuddlstm):
            decoder_lstm = CuDNNLSTM(self.decoder_latent_dim, return_sequences=True, return_state=True,
                                     name='rnn_decoder')  # returned state used in inference
        else:
            decoder_lstm = LSTM(self.decoder_latent_dim, return_sequences=True, return_state=True,
                                dropout=self.de_lstm_dropout, recurrent_dropout=self.de_lstm_recurent_dropout,
                                name='rnn_decoder')
        # decoder_outputs, _, _  = decoder_lstm(self.hared_embedding(decoder_inputs), initial_state=encoder_states)
        decoder_outputs, _, _ = decoder_lstm(self.shared_embedding(decoder_inputs),
                                             initial_state=encoder_model(encoder_inputs))
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='dcd_sfmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        decoder_teacher_forcing_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        ##################################################################
        ################# decoder-sampling model #########################
        ##################################################################
        # now for the SAMPLING models (re-arrangement of the prev one)
        # Remember that the training model varaibles were:
        #                                        decoder_outputs
        # encoder   --->    encoder_states  -->   decoder_lstm
        # shared_embedding                        shared_embeddings
        # encoder_inputs                          decdoer_inputs

        # the main difference is initial_state of decoder_lstm onces come from outside (encoder-result)
        # but then fed from the last cycle, thus we use Input c,h for it.
        # so they share their embedding, lstm and dense, but the input is configured to be more dynamic
        # it is intended to use this in a loop:
        # 1. h,c = encode sentence once , in_token= style_token
        # 2. loop:
        # 2.1      out_h,out_c,out_token = decoder_sampling_model(in_token,h,c)
        # 2.2      h = out_h, c=out_c , in_token = argmax(out_token)

        decoder_states_inputs = [Input(shape=(self.decoder_latent_dim,)), Input(shape=(self.decoder_latent_dim,))]

        # decoder_outputs2, state_h, state_c = decoder_lstm(self.shared_embedding(decoder_inputs), initial_state=decoder_states_inputs)
        decoder_outputs2, state_h, state_c = decoder_lstm(self.shared_embedding(decoder_inputs),
                                                          initial_state=decoder_states_inputs)

        decoder_outputs2 = decoder_dense(decoder_outputs2)

        decoder_sampling_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs2] + [state_h, state_c])

        return decoder_teacher_forcing_model, decoder_sampling_model

    def build_d(self):  # encoder_model,d__encoder_inputs):
        # IN on the sentnece embedding itself )list of h,c]
        # Out style embedding
        # d__states = encoder_model(d__encoder_inputs)

        d__states = [Input(shape=(self.decoder_latent_dim,)), Input(shape=(self.decoder_latent_dim,))]
        d__a = Concatenate(name='d__concat1')(d__states)
        d__a = Dropout(0.1, name='d__dropout1')(d__a)
        # if your inputs have shape  (batch_size, timesteps, features) and you want the dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features).
        d__a = Dense(100, activation='relu', name='d__dense1')(
            d__a)  # a = keras.layers.LeakyReLU()(a) #LeakyReLU  #why leaky? see: how to train your GAN - BUT IT FAILED TO LEARN (accuracy always 0.5)
        d__a = Dropout(0.1, name='d__dropout2')(d__a)
        d__style_outputs = Dense(self.style_out_size, activation='softmax', name='d__dense_softmax')(d__a)

        # d = Model(d_inputs,style_outputs)
        # d = Model(d__encoder_inputs,d__style_outputs)  #style_outputs : batch , one-hot-encoding-of-style
        d = Model(d__states, d__style_outputs,
                  name='styl_clsf')  # style_outputs : batch , one-hot-encoding-of-style
        return d

    def build_all(self):

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        encoder_model = self.build_encoder_model(encoder_inputs)
        model, decoder_sampling_model = self.build_decoder_model(encoder_model, encoder_inputs, decoder_inputs)
        # TODO: improve optimizer = ''#clipvalue=0.5,clipnorm=1.0)
        model.compile(optimizer='adam', loss='categorical_crossentropy')  # 30peocs loss: 0.2836 - val_loss: 0.4428

        # Seperate everything in d , both Encoder and Classfier
        d_encoder_model = self.build_encoder_model(encoder_inputs)
        d_encoder_model.trainable = False
        d_classifier_head = self.build_d()
        d = Model(encoder_inputs, d_classifier_head(d_encoder_model(encoder_inputs)))
        d.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # train_d(d,50) # TRAINING WELL alone , had used wrong names for models

        def inverse_categorical_crossentropy(y_true, y_pred):
            # need to implement it better , sum(1/categorical_crossentropy_per_sample)
            # if discriminator is random, on 2 styles, if expect 50% which should mean logloss of 1. so 1/1= 1
            # if discriminator is great, 99%, log-loss close to 0 , so 1/0 is big.
            # so expeceted range is GREAT=1 , BAD=BIGGG
            return 1 / (K.categorical_crossentropy(y_true, y_pred) + 0.0001)

        classifier_head = self.build_d()
        adv_d_out = classifier_head(encoder_model(encoder_inputs))  # encoder_model,encoder_inputs)
        g_d = Model([encoder_inputs, decoder_inputs], [model.output, adv_d_out])
        # in adv , encoder is not trainable. decoder is not.
        classifier_head.trainable = False

        g_d.compile(optimizer='adam',
                    loss=['categorical_crossentropy', inverse_categorical_crossentropy],
                    loss_weights=[1, self.adv_loss_weight])

        self.encoder_model = encoder_model
        self.decoder_sampling_model = decoder_sampling_model
        self.g = model

        self.d = d
        self.d_encoder_model = d_encoder_model
        self.d_classifier_head = d_classifier_head

        self.g_d = g_d
        self.classifier_head = classifier_head
        # self.encoder_model shared before

    def get_models(self):
        return [self.encoder_model, self.decoder_sampling_model, self.d, self.g, self.g_d]

    def get_model_names(self):
        return ['encoder_model', 'decoder_sampling_model', 'd', 'g', 'g_d']





class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = {'loss': [], 'val_loss': []}

    # def on_train_begin(self, logs={}):
    #  pass

    def on_epoch_end(self, batch, logs={}):
        for loss in ['loss', 'val_loss']:
            self.losses[loss].append(logs.get(loss))


class D_G_Trainer():
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.loss_history = LossHistory()
        self.loss_history_d = LossHistory()
        self.loss_history_adv = LossHistory()

    def train_g(self, steps, validation_steps=1, batch_size=64):
        self.model.g.fit_generator(self.dataset.gen_g(self.dataset.train, batch_size),
                                   steps,
                                   validation_steps=validation_steps,
                                   validation_data=self.dataset.gen_g(self.dataset.val, batch_size),
                                   callbacks=[self.loss_history],
                                   verbose=0 if steps < 10 else 1,
                                   max_queue_size=50,
                                   # workers=2
                                   )

    def train_g_d(self, steps, validation_steps=1, batch_size=64):
        self.model.classifier_head.set_weights(self.model.classifier_head.get_weights())

        self.model.g_d.fit_generator(self.dataset.gen_adv(self.dataset.train, batch_size),
                                     steps,
                                     validation_steps=validation_steps,
                                     validation_data=self.dataset.gen_adv(self.dataset.val, batch_size),
                                     max_queue_size=50,
                                     # workers=2,
                                     verbose=0 if steps < 20 else 1,
                                     callbacks=[self.loss_history_adv])

    def train_d(self, steps, validation_steps=1, batch_size=64):
        self.model.d_encoder_model.set_weights(self.model.encoder_model.get_weights())
        self.model.d.fit_generator(self.dataset.gen_d(self.dataset.train, batch_size),
                                   steps,
                                   validation_steps=validation_steps,
                                   validation_data=self.dataset.gen_d(self.dataset.val, batch_size),
                                   verbose=0 if steps < 20 else 1,
                                   max_queue_size=50,
                                   # workers=2,
                                   callbacks=[self.loss_history_d])
        print(self.loss_history_d)

    # summarize history for loss
    @staticmethod
    def plt_losses(l_h, title, with_val=False):
        print(l_h)

        plt.title(title)
        if len(l_h.losses['loss']) == 1:
            plt.scatter([0], l_h.losses['loss'][0], c='b')
            if with_val:
                plt.scatter([0], l_h.losses['val_loss'][0], c='g')
        elif len(l_h.losses['loss']) > 1:
            med = np.median(l_h.losses['loss'])
            plt.plot(l_h.losses['loss'][:])
            if with_val:
                plt.plot(l_h.losses['val_loss'][:])
            plt.ylim(ymin=-0.2)
            plt.ylim(ymax=med + 2.5)
        plt.ylabel('loss')
        plt.xlabel('batchse')
        plt.legend(['train', 'val'], loc='upper right')

    def plt_all(self, with_val=True):
        print(self.loss_history_d)
        plt.figure(figsize=(14, 4))
        plt.subplot(131)  # numrows, numcols, fignum
        D_G_Trainer.plt_losses(self.loss_history, 'g loss', with_val)
        plt.subplot(132)
        D_G_Trainer.plt_losses(self.loss_history_d, 'd loss', with_val)
        plt.subplot(133)
        D_G_Trainer.plt_losses(self.loss_history_adv, 'adv loss', with_val)
        plt.show()


def test():
    from dataset.bible import Num2WordsDataset
    dataset = Num2WordsDataset(start=1, end=10 * 1000 * 1000)

    model = D_G_Model(num_encoder_tokens=len(dataset.word2index),
                      num_decoder_tokens=len(dataset.word2index),  # from dataset 3628
                      style_out_size=len(dataset.style2index),  # from dataset 2
                      cuddlstm=False,
                      latent_dim=128,  # twice the default. make it stronger! but slower
                      bidi_encoder=True,
                      adv_loss_weight=1.0, )
    model.build_all()
    trainer = D_G_Trainer(model, dataset)
    trainer.train_g(10, 1)
    trainer.train_g_d(10, 1)
    trainer.train_d(10, 1)
    trainer.plt_all()


if __name__=='__main__':
    test()