import sys

from src.uni_embed.emb_utils import download_file_from_google_drive, EMBEDDINGS_GDRIVE_ID

LSTM_GDRIVE_ID = '1IKHJOeqkOzMEG3rSVJLFOhIky7ynSqk6'
TCNN_GDRIVE_ID = '1ofsHOPrxKtldw0-LgRhC_KnZ_0elybt9'

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: {} <lstm_dest_path> <tcnn_dest_path> <embeddings_dest_path>".format(sys.argv[0]))
    else:
        download_file_from_google_drive(LSTM_GDRIVE_ID, sys.argv[1])
        print("LSTM model downloaded as " + sys.argv[1])
        download_file_from_google_drive(TCNN_GDRIVE_ID, sys.argv[2])
        print("Temporal CNN model downloaded as " + sys.argv[2])
        download_file_from_google_drive(EMBEDDINGS_GDRIVE_ID, sys.argv[3])
        print("InferSent embeddings downloaded as " + sys.argv[3])
