import argparse
import random
import keras

from src.dataset.bible import BibleDataset
from src.uni_embed.emb_utils import emit_predictions, load_embeddings, emit_reference


def do_emit(message, model_path, output_path, samples):
    print(message)
    global dataset, embeddings, rng
    if dataset is None:
        print("Loading bible corpora...")
        dataset = BibleDataset(["bbe", "ylt"])
        rng = range(len(dataset.corpora['<ylt>']))
        if samples > 0:
            rng = sorted(random.sample(list(rng), samples))

    if model_path is not None:
        if embeddings is None:
            print("Loading sentence embeddings...")
            embeddings = load_embeddings(args.embeddings, ["bbe", "ylt"])
        print("Load model from {}...".format(model_path))
        model = keras.models.load_model(model_path)
        print("Emitting predictions for {} random samples...".format(len(rng)))
        corpus = emit_predictions(model, dataset, embeddings, rng)
    else:
        corpus = emit_reference(dataset, rng)

    with open(output_path, "w") as f:
        for s in corpus:
            print(" ".join(s), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emit gold corpus and TCNN and/or LSTM model predictions')
    parser.add_argument('--lstm', action="store", dest="lstm", help="path to save LSTM model predictions",
                        default=None)
    parser.add_argument('--tcnn', action="store", dest="tcnn", help="path to save TCNN model predictions",
                        default=None)
    parser.add_argument('--lstm-model', action="store", dest="lstm_model", help="path to LSTM model",
                        default="./exp/uni_embed/lstm/model.h5")
    parser.add_argument('--tcnn-model', action="store", dest="tcnn_model", help="path to TCNN model",
                        default="./exp/uni_embed/tcnn/model.h5")
    parser.add_argument('--gold', action="store", dest="gold", help="path to save gold corpus",
                        default=None)
    parser.add_argument('--samples', action="store", dest="samples",
                        help="Random samples count, 0 - calculate for the whole corpus (default)",
                        default="0")
    args = parser.parse_args()

    dataset = None
    embeddings = None

    args.samples = int(args.samples)

    if 'lstm' in args and args.lstm is not None:
        do_emit("Emitting LSTM model predictions", args.lstm_model, args.lstm, args.samples)

    if 'tcnn' in args and args.tcnn is not None:
        do_emit("Emitting TCNN model predictions", args.tcnn_model, args.tcnn, args.samples)

    if 'gold' in args and args.gold is not None:
        do_emit("Emitting gold corpus", None, args.gold, args.samples)
