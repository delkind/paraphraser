import numpy as np

def cycle( internal_gen, sampler, styles_indecies,end_symbol_index,max_sampled_sentence_len):
    """
    :param internal_gen: returning [x1,x2] and y
    :param sampler:
    :param styles_indecies: np.array with all styles, for example [11,12]
    :param end_symbol_index:  integer. for example 10
    :param max_sampled_sentence_len:
    :return:
    """

    while True:
        [x1,x2],y = next(internal_gen)

        # example [7,11,20] -> [0,1,2] -> randomize to  [2,0,1] -> map back to tokens
        #print('styles_indecies',styles_indecies)
        source_style_array = x2[:,0]
        #print (source_style_array)
        style2ordinal = { token:i for i,token in enumerate(styles_indecies)}
        #print(style2ordinal)
        ordinal2style = { i:token for i,token in enumerate(styles_indecies)}
        source_style_ordinals = np.array([style2ordinal[style] for style in source_style_array])
        # add to style 1..#styles then divide by #styles and keep reminder. example: #s =3  ,  (x + 1or2)%3
        num_of_styles = len(styles_indecies)
        random_style_oridinals = np.remainder(source_style_ordinals + np.random.randint(1, num_of_styles, size=(len(x2))),
                                         num_of_styles)
        #print (random_style_oridinals)
        random_style_tokens = np.array([ordinal2style[oridinal] for oridinal in random_style_oridinals])
        #print (random_style_tokens)

        sampled_input = sampler.decode_sequence_batch(x1, random_style_tokens, max_sampled_sentence_len , end_symbol_index)

        #print(x1.shape, x2.shape, y1.shape, 'max_sentence_length', self.max_sentence_length, np.max(x1), np.max(x2))
        yield [sampled_input,x2],y

'''
def gen_cycle_g(self, data_range, sampler, batch_size=64, noise_std=0.0):
    while True:
        # source_style_key= random.choices(list(self.corpora.keys()), k=1)[0]
        source_style_key = None
        target_style_key = random.choices(list(set(self.corpora.keys()) - {source_style_key}), k=1)[0]

        (batch, styles), (batch_noise, styles_noise) = self.noisy_sample_batch(data_range, batch_size=batch_size,
                                                                               one_style_only=source_style_key,
                                                                               noise_std=noise_std)
        # enc_input = self.enc_input(batch_noise)
        # print (batch_noise)
        enc_input_source = self.enc_input(batch)  # TODO:noise  #pad with <end>
        # = np.array(batch_noise)
        sampled_input = sampler.decode_sequence_batch(enc_input_source, self.word2index[target_style_key],
                                                      self.max_sentence_length + 1, self.word2index[self.end_symbol()],
                                                      verbose=False)
        # in: one-hundred and one <end>.  sampled to : 1 9 9
        # decoder: <wrd> one-hundred and one.  AND one-hot of one-hundred and one <end>

        # TODO: two current bugs: (64, 7) (64, 8) (64, 7, 146) max_sentence_length 7
        #                        (64, 7) (64, 6) (64, 7, 146) max_sentence_length 7
        # TODO: one-hot is of the decode with one-spare
        dec_input = self.dec_input(enc_input_source, [self.index2style[style] for style in styles])
        # print (batch) #end10 style11,12
        res = [enc_input_source, dec_input], to_categorical(enc_input_source, len(self.word2index)).astype(int)
        [x1, x2], y1 = res
        print(x1.shape, x2.shape, y1.shape, 'max_sentence_length', self.max_sentence_length, np.max(x1), np.max(x2))
        yield res


'''