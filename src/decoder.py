from termcolor import colored
import numpy as np


class SamplingDecoder():
    """
      init it with encoder_model and decoder_sampling_model
      It will be used to sample results(currently greedy)
    """

    def __init__(self, model_g_d, END_SYMBOL = '<end>'):
        self.model_g_d = model_g_d

    # TODO: now it's greedy and use argmax
    # maybe to choose randomly by distribution
    # and maybe to implement BEAM SEARCH
    def decode_sequence(self, input_seq, style_token, max_decoder_seq_length, end_symbol, verbose=False):
        """ input_seq as array of tokens of shape 1,SIZE why? TODO: fix
            style  as string  ('<bbe>')
        """
        if len(input_seq.shape) == 1:
            input_seq = input_seq.reshape(1, -1)  # now it's 1,N

        if verbose: print('input_seq', input_seq.shape, 'first 10', input_seq[0][:10])
        # Encode the input as state vectors.
        states_value = self.model_g_d.encoder_model.predict(input_seq)
        if verbose: print('encoder result states_value', 'h', states_value[0].shape, states_value[0].mean(), 'c',
                          states_value[1].shape, states_value[1].mean())

        # Generate empty target sequence of length 1.
        #                      batch,word-number value is token (0 /122)
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = style_token
        if verbose: print('target_seq', target_seq)

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            # start with encoder-state then change to self state
            output_tokens, h, c = self.model_g_d.decoder_sampling_model.predict([target_seq] + states_value)
            if verbose:
                print('output_tokens', output_tokens.shape, output_tokens.mean(), 'h', h.shape, 'c', c.shape)
                print('output softmax, first 10', output_tokens[0, 0, :10])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens)  # [0, -1, :])
            if verbose: print('sampled_token_index', sampled_token_index.shape, output_tokens.max(),
                              sampled_token_index)
            decoded_sentence.append(sampled_token_index)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_token_index == end_symbol or
                        len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        res = np.array(decoded_sentence)  # [dataset.index2word[index] for index in decoded_sentence]
        return res

    def show_sample(self, dataset, data_type='val', sample_ids=[], teacher_forcing=False):
        """
          data_type - train/val/test
          teacher_forcing : default False, sample argmax as for normal test. If true, feed decode-input from the dataset itself
          sample_ids : verse-id to sample (list)
          replace_style: if true, will pass a different style
        """
        styles = list(dataset.style2index.keys())
        # corp1 = dataset.corpora['<bbe>']
        # corp2 = dataset.corpora['<ylt>']
        # data1={'train': corp1[dataset.train[0]:dataset.train[1]], 'val':corp1[dataset.val[0]:dataset.val[1]] , 'test':corp1[dataset.test[0]:dataset.test[1]]}
        # data2={'train': corp2[dataset.train[0]:dataset.train[1]], 'val':corp2[dataset.val[0]:dataset.val[1]] , 'test':corp2[dataset.test[0]:dataset.test[1]]}

        try:

         for i in sample_ids:
            print('#' * 30, 'verb', i, '#' * 30)

            if data_type == 'train':
                r = dataset.train
            elif data_type == 'val':
                r = dataset.val
            else:
                r = dataset.test

            start, end = r[0], min(r[1], r[0] + 64)

            for style in styles:
                sentence = dataset.corpora[style][start:end][i]
                print(f'\n##encoder_input[{style}]:', dataset.recostruct_sentence(sentence))

                # decode it in every possible style, by replacing style
                one_x = dataset.enc_input([dataset.pad_sentence(sentence, dataset.max_sentence_length)]).reshape(1,
                                                                                                                 -1)  # model expect batch,N
                one_x_d = dataset.dec_input([dataset.pad_sentence(sentence, dataset.max_sentence_length)],
                                            [style]).reshape(1, -1)

                for replaced_style in styles:
                    # always replace_style:

                    gold_label = dataset.corpora[replaced_style][start:end][i]

                    if teacher_forcing:
                        one_x_d = dataset.dec_input([dataset.pad_sentence(gold_label, dataset.max_sentence_length)],
                                                    [replaced_style]).reshape(1, -1)
                        # print ('encoder get:',one_x[0][:10])
                        # print ('tf get:',one_x_d[0][:10] )

                        p = self.model_g_d.g.predict([one_x, one_x_d])
                        p = p[0].argmax(axis=1)
                        # print ('argmax:',p)
                        print(colored(f'decoder TF     [{replaced_style}]: {dataset.recostruct_sentence(p)}',
                                      'red'))  # skip first token of style
                    else:
                        # print (one_x)
                        max_decoder_seq_length = dataset.max_sentence_length + 1  # always pad one
                        p = self.decode_sequence(one_x, dataset.word2index[replaced_style], max_decoder_seq_length,dataset.word2index[dataset.end_symbol() ], verbose=False)
                        print(colored(f'decoder sample [{replaced_style}]: {dataset.recostruct_sentence(p)}','blue'))

                    print(f'gold label     [{replaced_style}]:', dataset.recostruct_sentence(gold_label))

        except :
            print('if the problem is cuda, make sure you are with GPU. if you changed to CPU, you must restart process in colab')