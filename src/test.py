from models import D_G_Model

from dataset.bible import Num2WordsDataset

dataset = Num2WordsDataset(start=1, end=10000000)

model = D_G_Model(num_encoder_tokens=len(dataset.word2index),
                      num_decoder_tokens=len(dataset.word2index),  #from dataset 3628
                      style_out_size=len(dataset.style2index), #from dataset 2
                      cuddlstm=False,
                      latent_dim = 512, #twice the default. make it stronger! but slower
                      bidi_encoder = True,
                      adv_loss_weight=1.0,)
model.build_all()

from decoder import SamplingDecoder
sampler= SamplingDecoder(model)
sampler.show_sample(dataset,'train',sample_ids=[0],teacher_forcing=True)

