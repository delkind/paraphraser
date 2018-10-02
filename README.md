# Sources for Sentence Paraphraser project

## Overview
The sources are intended to allow reproducing the experiments described in the project report
## Setting up
1. Clone the repository: <br>
`git clone https://github.com/delkind/paraphraser.git`<br>
`cd paraphraser`
1. Run setup script<br> 
`./setup.sh`
1. Please note that before executing any scripts as per instructions below, the following command should be invoked
to activate virtual environment:<br>
`source ./.env/bin/activate`
 
## Pre-trained universal embeddings (InferSent) experiment
 
### Creating the paraphrases of the bible BBE dataset
1. Download the pre-trained TCNN and LSTM based decoders and pre-built universal embeddings for Bible dataset by running<br>
`./dl_uni_emb_files.sh`
1. To calculate BLEU score for both models for _n_ random samples please run<br>
`./uni_emb_calc_bleu.sh --samples <n>`
1. To emit the original sentences (GOLD) file please run<br>
`./uni_emb_create_gold.sh`
1. To emit LSTM model predictions file please run<br>
`./uni_emb_lstm_pred.sh`
1. To emit TCNN model predictions file please run<br>
`./uni_emb_tcnn_pred.sh`

### Re-building experiment models and embeddings

The instructions above assume usage of pre-trained models and pre-built embeddings in order to produce the predictions 
and evaluate the experiment results. Below we provide the instructions for re-building and re-training models and 
embeddings instead of using the pre-built ones.  

#### Reproducing sentence embedding creation
1. Setup InferSent data files by running<br>
`./setup_infersent.sh`
1. Install PyTorch - follow the instructions [here](https://pytorch.org/get-started/locally/). 
We haven't provided a script since the installation differs substantially depending on the platform. 
1. Create embeddings from the YLT and BBE bible corpora by running<br>
`./create_uni_emb.sh`
1. Verify that `exp/uni_embed/embeddings.h5` file is created

#### Reproducing models training
We have experimented with the decoder based on LSTM and Temporal CNN architectures. To train the LSTM-based decoder
run<br>
`./uni_emb_train_lstm.sh`
<br>To train the TCNN-based decoder run <br>
`./uni_emb_train_tcnn.sh`
 <br>In order to specify the number of epochs for training `--epochs <n>` parameter can be specified to both scripts
 where _n_ is the number of epochs. The default is to train for 10 epochs. The model is saved (and subsequently overwritten) 
 after each epoch.
