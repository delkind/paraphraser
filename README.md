# Sources for Sentence Paraphraser project

## Overview
The sources are intended to allow reproducing the experiments described in the project report
## Setting up
1. Clone the repository: <br>
`git clone https://github.com/delkind/paraphraser.git`<br>
`cd paraphraser`
2. Run setup script<br> 
`./setup.sh`
 
 ## Universal embeddings experiment
 
 ### Reproducing sentence embedding creation
 1. Create embeddings from the YLT and BBE bible corpora by running<br>
 `./create_uni_emb.sh`
 2. Verify that `exp/uni_embed/embeddings.h5` file is created
 3. Alternatively, you can download the prebuilt embeddings file by running<br>
 `./dl_uni_emb_files.sh`
 
 ### Creating the paraphrases of the bible BBE dataset
 1. Download the pre-trained TCNN and LSTM based decoders and universal embeddings for Bible dataset by running<br>
 `./dl_uni_emb_files.sh`
 
 ### Reproducing models training
 We have experimented with the decoder based on LSTM and Temporal CNN architectures. To train the LSTM-based decoder
 run<br>
 `./uni_emb_train_lstm.sh`
 To train the TCNN-based decoder run <br>
 `./uni_emb_train_tcnn.sh`
 