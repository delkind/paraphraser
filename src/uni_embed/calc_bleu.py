import argparse
import random

import keras

from src.dataset.bible import BibleDataset
from src.uni_embed.emb_utils import load_embeddings, emit_predictions, emit_reference


def calculate_bleu_score(lstm_path, tcnn_path, emb_path, samples):
    print("Loading corpora...")
    dataset = BibleDataset(["bbe", "ylt"])
    rng = range(len(dataset.corpora['<ylt>']))
    if samples > 0:
        rng = random.sample(list(rng), samples)
    print("Calculating BLEU scores for {} random samples...".format(len(rng)))
    print("Loading pre-built sentence embeddings...")
    embeddings = load_embeddings(emb_path, ["bbe", "ylt"])
    print("Loading LSTM model...")
    model = keras.models.load_model(lstm_path)
    print("Emitting LSTM model predictions...")
    h = emit_predictions(model, dataset, embeddings, rng)
    r = [[s] for s in emit_reference(dataset, rng)]
    from nltk.translate.bleu_score import corpus_bleu
    lstm_scores = {
        'BLEU-1': corpus_bleu(r, h, (1, 0, 0, 0)),
        'BLEU-2': corpus_bleu(r, h, (0.5, 0.5, 0, 0)),
        'BLEU-3': corpus_bleu(r, h, (0.33, 0.33, 0.33, 0)),
        'BLEU-4': corpus_bleu(r, h)}
    print("Loading TCNN model...")
    model = keras.models.load_model(tcnn_path)
    print("Emitting TCNN model predictions...")
    h = emit_predictions(model, dataset, embeddings, rng)
    return {
        "LSTM": lstm_scores,
        "TCNN": {
            'BLEU-1': corpus_bleu(r, h, (1, 0, 0, 0)),
            'BLEU-2': corpus_bleu(r, h, (0.5, 0.5, 0, 0)),
            'BLEU-3': corpus_bleu(r, h, (0.33, 0.33, 0.33, 0)),
            'BLEU-4': corpus_bleu(r, h)
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate BLEU scores for LSTM and/or TCNN models')
    parser.add_argument('--lstm', action="store", dest="lstm", help="path to pretrained LSTM model",
                        default="./exp/uni_embed/lstm/model.h5")
    parser.add_argument('--tcnn', action="store", dest="tcnn", help="path to pretrained TCNN model",
                        default="./exp/uni_embed/tcnn/model.h5")
    parser.add_argument('--embeddings', action="store", dest="embeddings", help="path to bible embeddings",
                        default="./exp/uni_embed/embeddings.h5")
    parser.add_argument('--samples', action="store", dest="samples",
                        help="Random samples count, 0 - calculate for the whole corpus (default)",
                        default="0")
    args = parser.parse_args()
    scores = calculate_bleu_score(args.lstm, args.tcnn, args.embeddings, int(args.samples))
    for (k, v) in scores.items():
        print("{}:".format(k))
        for (vk, vv) in v.items():
            print("\t{}: {:.3f}".format(vk, vv))
