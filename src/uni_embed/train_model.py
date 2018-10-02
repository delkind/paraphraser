import argparse

from src.dataset.bible import BibleDataset
from src.uni_embed.emb_utils import train_model, lstm_model, load_embeddings, tcnn_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM and/or TCNN models')
    parser.add_argument('--lstm', action="store", dest="lstm", help="path to save LSTM model",
                        default=None)
    parser.add_argument('--tcnn', action="store", dest="tcnn", help="path to save TCNN model",
                        default=None)
    parser.add_argument('--epochs', action="store", dest="epochs",
                        help="Number of epochs for training (default is 10)",
                        default="10")
    parser.add_argument('--embeddings', action="store", dest="embeddings", help="path to bible embeddings",
                        default="./exp/uni_embed/embeddings.h5")
    args = parser.parse_args()

    if 'lstm' in args and args.lstm is not None:
        dataset = BibleDataset(["bbe", "ylt"])
        embeddings = load_embeddings(args.embeddings, ["bbe", "ylt"])
        print("Training LSTM model...")
        train_model(dataset, embeddings, lstm_model, int(args.epochs), args.lstm)
    else:
        dataset = None
        embeddings = None

    if 'tcnn' in args and args.tcnn is not None:
        if dataset is None:
            dataset = BibleDataset(["bbe", "ylt"])
        if embeddings is None:
            embeddings = load_embeddings(args.embeddings, ["bbe", "ylt"])
        print("Training TCNN model...")
        train_model(dataset, embeddings, tcnn_model, int(args.epochs), args.tcnn)
